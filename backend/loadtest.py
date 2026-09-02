"""Drives concurrent simulated visitors against a running instance of the
app (loadtest_server.py, or any instance with Suno/Anthropic calls already
stubbed) and reports latency/error-rate/memory as concurrency scales.

Each simulated visitor is its own thread with its own cookie, doing
repeated tag -> generate cycles -- deliberately not using requests'
automatic cookie jar, which enforces the Secure flag main.py's session
cookie carries the same way a browser would (refusing to send it back over
plain http); the session_id value is captured from the first response and
passed back explicitly instead.

Sample images only (never a real upload) and judges=false -- this is about
measuring the backend's own concurrent-request capacity, not re-testing the
tagging pipeline or paying for Anthropic calls that have nothing to do with
Cloud Run sizing.

Usage: python3 loadtest.py --url http://127.0.0.1:8098 --concurrency 20 --cycles 3 [--pid 12345]
"""

import argparse
import statistics
import threading
import time

import requests

SAMPLE_IMAGES = [
    "01-nature.png", "02-night-sky.png", "03-portrait-hooded.png", "04-ai-generated-1.png",
    "05-minion.png", "06-cool-cat.png", "07-spooky.png", "08-mona-lisa.png",
]

# How often to re-poll GET /generate/{run_id}/status while queued. Faster
# than the frontend's own ~3s cadence -- there's no real client here to be
# gentle with, and a tighter loop gets more position samples per run to
# verify the countdown against.
QUEUE_POLL_SECONDS = 1.0


class VisitorResult:
    def __init__(self):
        self.latencies: dict[str, list[float]] = {"tag": [], "generate": []}
        self.errors: list[str] = []
        # How many /generate calls came back `queued` at least once, out of
        # how many were attempted -- lets main() report what fraction of
        # this run actually exercised the queue rather than only the fast
        # path (which needs concurrency > MAX_CONCURRENT_GENERATIONS to
        # ever happen at all).
        self.queued = 0
        self.generate_attempts = 0


def run_visitor(
    base_url: str, cycles: int, visitor_index: int, result: VisitorResult, skip_generate: bool = False
) -> None:
    session_cookie: str | None = None
    image = SAMPLE_IMAGES[visitor_index % len(SAMPLE_IMAGES)]

    for cycle in range(cycles):
        cookies = {"session_id": session_cookie} if session_cookie else {}

        start = time.monotonic()
        try:
            resp = requests.post(
                f"{base_url}/tag",
                params={"algorithm": "siglip", "judges": "false", "sample_image": image},
                cookies=cookies,
                timeout=30,
            )
        except requests.RequestException as exc:
            result.errors.append(f"tag transport error: {exc}")
            continue
        result.latencies["tag"].append(time.monotonic() - start)

        if session_cookie is None:
            session_cookie = resp.cookies.get("session_id")
        if resp.status_code != 200:
            result.errors.append(f"tag HTTP {resp.status_code}: {resp.text[:200]}")
            continue
        run_id = resp.json()["id"]

        if skip_generate:
            continue

        result.generate_attempts += 1
        start = time.monotonic()
        try:
            resp = requests.post(
                f"{base_url}/generate",
                params={"run_id": run_id},
                cookies={"session_id": session_cookie},
                timeout=30,
            )
        except requests.RequestException as exc:
            result.errors.append(f"generate transport error: {exc}")
            continue
        if resp.status_code != 200:
            result.errors.append(f"generate HTTP {resp.status_code}: {resp.text[:200]}")
            continue

        # A `queued` POST response means apiframe's MAX_CONCURRENT_GENERATIONS
        # slots were all taken -- poll GET .../status until this run finishes
        # (or errors), so the latency recorded below reflects total
        # wait-plus-render time, not just the near-instant initial POST. A
        # fast-path response (a slot was free) has no "status" key at all --
        # it's the same finished-song shape /generate has always returned --
        # so the loop below never runs for that case.
        body = resp.json()
        positions: list[int] = []
        was_queued = False
        while body.get("status") == "queued":
            was_queued = True
            if "position" in body:
                positions.append(body["position"])
            time.sleep(QUEUE_POLL_SECONDS)
            try:
                resp = requests.get(
                    f"{base_url}/generate/{run_id}/status",
                    cookies={"session_id": session_cookie},
                    timeout=30,
                )
            except requests.RequestException as exc:
                result.errors.append(f"generate status transport error: {exc}")
                body = {}
                break
            if resp.status_code != 200:
                result.errors.append(f"generate status HTTP {resp.status_code}: {resp.text[:200]}")
                body = {}
                break
            body = resp.json()

        result.latencies["generate"].append(time.monotonic() - start)
        result.queued += was_queued

        if len(positions) > 1 and any(b > a for a, b in zip(positions, positions[1:])):
            result.errors.append(f"queue position increased instead of counting down: {positions}")
        if body.get("status") == "error":
            result.errors.append(f"generate error: {body.get('detail')}")


def sample_memory_rss(pid: int, samples: list[int], stop: threading.Event) -> None:
    try:
        import psutil
    except ImportError:
        return
    process = psutil.Process(pid)
    while not stop.is_set():
        try:
            samples.append(process.memory_info().rss)
        except psutil.NoSuchProcess:
            return
        stop.wait(0.5)


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(len(ordered) * p))
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8098")
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--cycles", type=int, default=3, help="tag+generate cycles per visitor")
    parser.add_argument("--pid", type=int, default=None, help="server PID, for memory sampling (needs psutil)")
    parser.add_argument(
        "--skip-generate", action="store_true",
        help="tag only -- for testing against a real (unmocked) instance where /generate would need a real APIFRAME_API_KEY",
    )
    parser.add_argument(
        "--ramp-seconds", type=float, default=0,
        help="spread visitor starts evenly over this many seconds instead of firing them all at once "
             "-- 25 users arriving over 20s is a very different load shape than 25 arriving in the same instant",
    )
    args = parser.parse_args()

    results = [VisitorResult() for _ in range(args.concurrency)]

    mem_samples: list[int] = []
    stop_sampling = threading.Event()
    mem_thread = None
    if args.pid:
        mem_thread = threading.Thread(target=sample_memory_rss, args=(args.pid, mem_samples, stop_sampling))
        mem_thread.start()

    wall_start = time.monotonic()
    threads = [
        threading.Thread(
            target=run_visitor, args=(args.url, args.cycles, i, results[i], args.skip_generate)
        )
        for i in range(args.concurrency)
    ]
    stagger = (args.ramp_seconds / args.concurrency) if args.ramp_seconds and args.concurrency > 1 else 0
    for t in threads:
        t.start()
        if stagger:
            time.sleep(stagger)
    for t in threads:
        t.join()
    wall_elapsed = time.monotonic() - wall_start

    stop_sampling.set()
    if mem_thread:
        mem_thread.join(timeout=2)

    all_tag = [v for r in results for v in r.latencies["tag"]]
    all_generate = [v for r in results for v in r.latencies["generate"]]
    all_errors = [e for r in results for e in r.errors]
    total_requests = len(all_tag) + len(all_generate) + len(all_errors)
    total_generate_attempts = sum(r.generate_attempts for r in results)
    total_queued = sum(r.queued for r in results)
    cap_violations = [e for e in all_errors if "concurrency cap violated" in e]

    print(f"\n=== concurrency={args.concurrency} cycles={args.cycles} wall_time={wall_elapsed:.1f}s ===")
    print(f"total requests attempted: {total_requests}, errors: {len(all_errors)} "
          f"({100 * len(all_errors) / max(total_requests, 1):.1f}%)")
    if not args.skip_generate:
        print(
            f"generate calls that hit the queue (not the fast path): {total_queued}/"
            f"{total_generate_attempts}"
        )
        print(
            "apiframe concurrency cap: "
            + (f"VIOLATED {len(cap_violations)} time(s) -- see errors below" if cap_violations else "never violated")
        )
    for label, values in [("tag", all_tag), ("generate", all_generate)]:
        if not values:
            print(f"{label}: no successful requests")
            continue
        print(
            f"{label}: n={len(values)} "
            f"p50={percentile(values, 0.50):.2f}s p95={percentile(values, 0.95):.2f}s "
            f"p99={percentile(values, 0.99):.2f}s max={max(values):.2f}s"
        )
    if all_errors:
        print("first 5 errors:")
        for e in all_errors[:5]:
            print(f"  {e}")
    if mem_samples:
        print(
            f"server RSS during run: min={min(mem_samples) / 1e6:.0f}MB "
            f"max={max(mem_samples) / 1e6:.0f}MB avg={statistics.mean(mem_samples) / 1e6:.0f}MB"
        )
    elif args.pid:
        print("no memory samples collected (is psutil installed? `pip install psutil`)")


if __name__ == "__main__":
    main()
