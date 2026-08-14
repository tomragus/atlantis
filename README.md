# Image to Song

Upload a picture and (optionally) spin a genre knob; the backend embeds the
image in SigLIP space, scores it against a fixed adjective vocabulary to pull
its top 10 tags, composes those tags with the genre into a prompt, and sends
that prompt to Suno (via apiframe) to render a track. Both takes are saved to
local history, where they're viewable and re-listenable alongside the tags
and scores that produced them.

This repo stops at song generation by design — no cover art, no visualizer,
no export, no publishing to social platforms. It's a clean testbed for
iterating on the tagging/prompting pipeline, not the full pipeline itself.

## Running it locally

### Backend

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate   # needs Python 3.10+
pip install -r requirements.txt
cp .env.example .env   # then paste your APIFRAME_API_KEY into .env
uvicorn main:app --reload
```

Runs on `http://127.0.0.1:8000`. `GET /` reports whether the key loaded.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Runs on `http://localhost:3000` and talks to the backend at
`http://127.0.0.1:8000` by default (override with `NEXT_PUBLIC_API_URL`).

Both servers need to be running for the app to work.
