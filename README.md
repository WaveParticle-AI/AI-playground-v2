# Waveparticle Playground

A standalone Streamlit web playground for testing AI characters across multiple
LLM providers (OpenAI, DeepSeek, Gemini, Qwen) with full visibility into the
reasoning pipeline and RAG retrievals.

## What this is

A side-by-side comparison harness for character-driven LLM prompts. Pick a
character (Oppenheimer ships by default), pick up to four models from across
providers, send a single user message, and see each model's final reply,
its raw thinking content, the RAG chunks it received, the mood classification
that shaped the prompt, and the fully assembled message list — all in one
view.

## BYOK — Bring Your Own Key

This app stores **no** credentials server-side. You paste each provider's API
key into the sidebar; keys live only in `st.session_state` for the duration of
your browser session and are sent only to the corresponding provider's API.
There is no telemetry, no logging of keys, and no persistence to disk.

## Run locally

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`), paste at
least one provider key in the sidebar, pick a model, and send a message.

## Add a new character

Drop a folder under `characters/`:

```
characters/<your_character_id>/
├── persona.md           # the system prompt body
├── voice_reminder.md    # short binding voice anchor
└── knowledge/           # optional, used by RAG
    ├── 01_topic.md
    └── 02_topic.md
```

The folder name becomes the character `id`. Knowledge markdown files are
chunked by H2 (`## ...`) headers; each H2 section becomes a retrievable chunk.

## Deploy to Render

This repo includes a `render.yaml` Blueprint targeting Render's free tier.
Push to GitHub, connect the repo as a Render Blueprint, and Render will build
and deploy automatically. Because the app is BYOK, no environment variables
need to be configured server-side.

## License

MIT — see `LICENSE`.
