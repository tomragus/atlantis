"""Generates eval/report.html from eval/results.json.

Self-contained: every image is embedded as a base64 JPEG thumbnail directly in the
HTML, so the report opens in a browser with no server and no external assets.

Can be regenerated standalone (without re-running the judge) via:
    backend/.venv/bin/python -c "
import json
from pathlib import Path
from report import generate_report
results = json.loads(Path('results.json').read_text())
generate_report(results, Path('../test-images'), Path('report.html'))
"
"""

import base64
import html
import io
import json
from pathlib import Path

from PIL import Image

THUMBNAIL_MAX_SIDE = 320


def _thumbnail_b64(image_path: Path) -> str:
    image = Image.open(image_path).convert("RGB")
    image.thumbnail((THUMBNAIL_MAX_SIDE, THUMBNAIL_MAX_SIDE), Image.LANCZOS)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _tag_html(tag: dict) -> str:
    word = html.escape(tag["tag"])
    score = tag["score"]
    if tag["flagged"]:
        reason = html.escape(tag["reason"])
        return (
            f'<span class="tag flagged" title="{reason}">'
            f'{word} <span class="score">{score:.3f}</span>'
            f'<div class="reason">{reason}</div>'
            f"</span>"
        )
    return f'<span class="tag">{word} <span class="score">{score:.3f}</span></span>'


def _card_html(entry: dict, images_dir: Path) -> str:
    filename = entry["filename"]
    if "error" in entry:
        return (
            '<div class="card error">'
            f"<h3>{html.escape(filename)}</h3>"
            f'<p class="error-msg">Failed: {html.escape(entry["error"])}</p>'
            "</div>"
        )

    image_path = images_dir / filename
    try:
        thumb_b64 = _thumbnail_b64(image_path)
        img_html = f'<img src="data:image/jpeg;base64,{thumb_b64}" alt="{html.escape(filename)}">'
    except Exception as exc:
        img_html = f'<p class="error-msg">thumbnail failed: {html.escape(str(exc))}</p>'

    n_flagged = sum(1 for t in entry["tags"] if t["flagged"])
    tags_html = "\n".join(_tag_html(t) for t in entry["tags"])
    flagged_badge = (
        f'<span class="badge">{n_flagged} flagged</span>' if n_flagged else ""
    )

    return f"""
<div class="card">
  <div class="thumb">{img_html}</div>
  <div class="details">
    <h3>{html.escape(filename)} {flagged_badge}</h3>
    <div class="tags">{tags_html}</div>
  </div>
</div>"""


CSS = """
:root { color-scheme: light dark; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  margin: 0; padding: 24px; background: #f5f5f7; color: #1a1a1a;
}
h1 { font-size: 20px; margin-bottom: 4px; }
.subtitle { color: #666; margin-bottom: 24px; font-size: 14px; }
.grid { display: flex; flex-direction: column; gap: 16px; }
.card {
  display: flex; gap: 20px; background: white; border-radius: 10px;
  padding: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}
.card.error { border-left: 4px solid #d33; }
.thumb img {
  width: 220px; height: 220px; object-fit: cover; border-radius: 6px; flex-shrink: 0;
}
.details { flex: 1; min-width: 0; }
h3 { margin: 0 0 12px 0; font-size: 15px; }
.badge {
  background: #d33; color: white; font-size: 11px; padding: 2px 8px;
  border-radius: 10px; font-weight: 600; vertical-align: middle;
}
.tags { display: flex; flex-wrap: wrap; gap: 8px; }
.tag {
  position: relative; background: #eef0f3; border-radius: 6px; padding: 5px 9px;
  font-size: 13px; display: inline-flex; align-items: baseline; gap: 5px;
}
.tag .score { font-size: 11px; color: #888; }
.tag.flagged {
  background: #fdecea; border: 1px solid #e8a5a0; color: #8a1f14; cursor: help;
}
.tag.flagged .score { color: #b0554a; }
.tag.flagged .reason { display: none; }
.tag.flagged:hover .reason {
  display: block; position: absolute; top: 110%; left: 0; z-index: 10;
  background: #2a2a2a; color: white; padding: 8px 10px; border-radius: 6px;
  font-size: 12px; width: 240px; white-space: normal; box-shadow: 0 4px 12px rgba(0,0,0,0.3);
}
.error-msg { color: #d33; font-size: 13px; }
@media (prefers-color-scheme: dark) {
  body { background: #1c1c1e; color: #eee; }
  .card { background: #2c2c2e; box-shadow: none; }
  .tag { background: #3a3a3c; }
  .tag .score { color: #aaa; }
  .tag.flagged { background: #4a2320; border-color: #7a3a32; color: #ffb4a8; }
  .subtitle { color: #999; }
}
"""


def generate_report(results: list[dict], images_dir: Path, output_path: Path) -> None:
    total_images = len(results)
    total_flagged = sum(
        1 for r in results for t in r.get("tags", []) if t["flagged"]
    )
    cards = "\n".join(_card_html(entry, images_dir) for entry in results)

    html_doc = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>SigLIP tag judge report</title>
<style>{CSS}</style>
</head>
<body>
<h1>SigLIP tag judge report</h1>
<p class="subtitle">
  {total_images} images, judged by Claude Sonnet 5.
  {total_flagged} tag flags total. Hover a flagged tag for the judge's reason.
</p>
<div class="grid">
{cards}
</div>
</body>
</html>"""

    output_path.write_text(html_doc)


if __name__ == "__main__":
    results_path = Path(__file__).parent / "results.json"
    images_dir = Path(__file__).parent.parent / "test-images"
    output_path = Path(__file__).parent / "report.html"
    generate_report(json.loads(results_path.read_text()), images_dir, output_path)
