# Ideagram Studio

Ideagram Studio: A better way to compose mood-accurate songs from images using joint embedding spaces

### [Live Site](https://ideagram.studio/)

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

Runs on `http://127.0.0.1:8000`. `GET /api/health` reports whether each key loaded.

### Frontend

```bash
cd frontend
npm install
npm run dev:proxied   # not plain `npm run dev` -- see below
npm run dev:proxy     # in a second terminal
```

Open **`http://127.0.0.1:3001`** (the proxy), not `:3000` directly. The app
now identifies each visitor with a `session_id` cookie, and the browser
treats `localhost:3000`/`127.0.0.1:8000` as different sites -- a cookie set
by one is never sent back to the other, so every request looks like a brand
new, empty visitor and `/generate` 404s on any run you just tagged.
`dev-proxy.js` puts both servers behind one origin, the same way production
already does (one process serving frontend + API together). Plain
`npm run dev` still works for frontend-only changes that never call the
backend, but skips the proxy and inherits that broken-cookie behavior.

Both the backend and (both halves of) the frontend need to be running for
the full app to work.
