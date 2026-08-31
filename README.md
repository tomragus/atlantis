# Ideagram Studio

Ideagram Studio: A better way to compose mood-accurate songs from images using joint embedding spaces

Some backstory: I got REALLY into Suno this year. We all know about AI slop, and yes, the internet is filled with it. When I first heard about Suno, I was sure it would just be another slop machine, maybe fun as a gimmick but nothing to take too seriously. But when I first tried it out, I was blown away by just how *musical* the outputs were. And since Suno's freemium model is incredibly generous towards staunch non-payers such as myself (usually offering 20+ song generations per day for free), I found myself constantly hopping on to blow off some steam by making fun AI tunes. 🎶

This summer I took a trip to San Francisco and I thought it would be really fun if I could make soundtracks for some of the pictures I was taking. Unfortunately, Suno doesn't accept image inputs, so I tried a couple workarounds. First I tried Google Lyria 3 which does accept images, but the songs that came out the other end never sounded anything like the pictures. They would generally be songs *about* the pictures, in the sense that, the lyrics would be about what is in the image. But they didn't *sound* anything like the pictures the way a soundtrack would. No matter how I'd prompt it, it didn't seem like the Lyria model even understood what I meant by an image's "mood". Pretty un-musical for a music model, if you ask me.

Next I tried uploading the image to Claude first with the prompt "Give me 10 descriptive adjective tags for this image, in a list" (I learned from my experimentation that Suno responds very well to descriptive adjective tags), and then fed that into Suno. This produced much more interesting results. Some images were better than others, and I'd say about 60% of the tracks still didn't really sound much like the pictures they came from- but every once in a while, the flow would really nail it. The first image -> song transformation that really caught my attention, and kind of kicked this whole thing off, was this: 

<img width="3024" height="4032" alt="IMG_1434" src="https://github.com/user-attachments/assets/6f93e8d3-633f-4fa2-bb78-6e890aad5fd7" />

Yup- my messy hotel room, with my nasty stinky foot sitting right on the desk (after walking 13 miles that day). The song that Suno made was this zany, chipmunk-style track that, to me, captures the strangeness and haphazard-ness of this picture near-perfectly. Have a listen:

>track here<

Wild, right? I unfortunately wasn't able to reproduce this with subsequent tries, and it was likely a stroke of weird luck, but this was kind of eye-opening for me. It showed that at least in theory (this whole "a song that sounds like the image" thing was possible (??) )

When I got back home I thought it would be fun to build a little applet where I could just upload an image directly and have a Claude API do the intermediate "tag extraction" step for me so I could quickly experiment with more images. 


I had just recently given a talk on Chapter 3 of Chip Huyen's "AI Engineering", and one of the topics in the chapter was evaluating semantic similarity between texts (and/or other modalities) using cosine similarity of embedding vectors (or, the distance between those vectors in the joint embedding space). One day the idea crossed my mind- "what if instead of making AI write tags, I embed both the image and some vocabulary and see which vocabulary words show up close to the image in the joint embedding space?"

(concern- too low complexity)
So, I threw together some vocabulary [which was edited and optimized later] 




TOO MESSY ; TOO EDITORIAL ; NOT ACADEMIC ENOUGH ??















Since then, I've tried many other audio generation models: facebook's local musicgen models, Minimax models, Google Lyria 3... they're all capable at audio generation and will do what you ask, but they just aren't *musical* in the way the Suno models are. 





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
