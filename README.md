# Lexis

Lexis is a full-stack editorial rewriting app for prose. You paste text, choose a model, and receive a streamed rewrite designed to preserve meaning while improving rhythm, imagery, and style.

## What It Does

- Rewrites prose with a literary/editorial system prompt
- Supports both OpenAI and Anthropic providers
- Streams rewrite results from the backend to the UI
- Highlights changed phrases so edits can be reviewed in context
- Generates alternative phrasings for individual replacements
- Lets users toggle changes, edit the final rewrite, and use undo/redo
- Streams the proof into view as the model writes it
- Stores theme, provider, model, effort, style directive, and browser-side API keys locally
- Folio mode (pro): an endless sheet for long documents, Markdown import and export, and paragraphs proofed live as you type

## Folio Mode

Switch **Mode** in the masthead from Desk to Folio. Desk sends one passage when you ask; Folio holds a whole document and works it paragraph by paragraph.

- **Writing.** Each paragraph is its own block. Enter starts a new one, Shift + Enter breaks a line, Backspace at the start of a paragraph joins it to the one above, and the arrow keys move between paragraphs.
- **Live proofing.** With Live on, a paragraph goes to the editor about a second after you stop typing, or as soon as you leave it. Its proof streams into the right column beside it. Editing a paragraph cancels its request in flight and dims its old proof until the new one lands. Ctrl/Cmd + Enter proofs the current paragraph right away.
- **Context.** Each request carries the preceding text (up to 1,500 characters) and the next paragraph, so the editor keeps voice and continuity without rewriting the neighbours.
- **Corpus import.** Import .md files or a whole folder, or drop files on the sheet. Files are read in natural order (`ch2` before `ch10`), YAML front matter is dropped, headings, rules, and fenced code stay verbatim and are never sent, and each file is labelled where it starts. Imported text is not sent automatically: Proof all queues every unproofed paragraph, three at a time, and asks first when there are more than 25.
- **Reviewing.** Click a red mark to restore the original wording. Accept into manuscript replaces the paragraph with its proof; Keep original exports the manuscript instead of the proof.
- **Export.** Export .md downloads the document with proofs applied, restorations and kept paragraphs honoured, and stale proofs skipped in favour of the newer manuscript.
- **Limits.** Proofs are cached per paragraph, so unchanged text is never sent twice in a session. A rate-limited paragraph pauses the queue and retries up to three times; a key error stops the queue. The folio is saved in local storage, which holds a few megabytes; past that Lexis says so and you should export.

## Stack

- Frontend: React 18, TypeScript, Vite, Tailwind utilities
- Backend: FastAPI, Python 3.12
- AI providers: OpenAI and Anthropic
- Local tooling: `bun` for frontend dependencies, `uv` for Python environment management

## Repository Layout

```text
.
|- backend/
|  |- app/main.py        # FastAPI app and streaming endpoints
|  |- pyproject.toml     # Python package metadata and dependencies
|- frontend/
|  |- src/App.tsx        # Main single-page application (Desk mode)
|  |- src/Folio.tsx      # Folio mode: block editor and live proofing queue
|  |- src/lib/           # Shared proof helpers and the Markdown splitter
|  |- package.json       # Frontend scripts and dependencies
|- start.sh              # Starts backend and frontend together
|- vercel.json           # Vercel multi-service configuration
```

## Requirements

- Python 3.12+
- `uv`
- `bun`
- At least one provider API key:
  - `OPENAI_API_KEY`, or
  - `ANTHROPIC_API_KEY`

## Quick Start

1. Create a root `.env` file with the provider key(s) you want to use.
2. Run:

```bash
./start.sh
```

This script will:

- create `backend/.venv` if needed
- install backend dependencies with `uv`
- install frontend dependencies with `bun` if `node_modules` is missing
- start the API on `http://127.0.0.1:8000`
- start the frontend on `http://127.0.0.1:5173`

## Environment Variables

Server-side configuration is read via `python-dotenv`.

```env
OPENAI_API_KEY=
ANTHROPIC_API_KEY=

DEFAULT_PROVIDER=openai
DEFAULT_MODEL=
DEFAULT_OPENAI_MODEL=gpt-6.1-sol
DEFAULT_ANTHROPIC_MODEL=claude-opus-5-5
```

Notes:

- Available models: OpenAI `gpt-6.1-sol`, `gpt-6-astra`, `gpt-5.6-terra`, `gpt-6-luna`; Anthropic `claude-opus-5-5`, `claude-fable-5-1`, `claude-sonnet-5-5`, `claude-haiku-5-5`.
- The frontend can also send an API key directly per request from the browser.
- Browser-entered keys are stored locally in local storage.
- If no browser key is set, the backend falls back to environment variables.

## Manual Development

Backend:

```bash
uv venv
uv pip install --python .venv/bin/python -e .
./.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Run those commands in `backend/`.

Frontend:

```bash
bun install
VITE_API_URL="http://127.0.0.1:8000" bun run dev --host 127.0.0.1 --port 5173
```

Run those commands in `frontend/`.

## API

### `GET /healthz`

Returns a basic health response and the resolved default provider.

### `POST /analyze`

Streams a rewrite response as server-sent events.

Request body:

```json
{
  "text": "Your source text",
  "style_directive": "optional style guidance",
  "provider": "openai",
  "model": "gpt-6.1-sol",
  "api_key": "optional override",
  "effort": "medium",
  "context_before": "optional preceding text",
  "context_after": "optional following text"
}
```

`context_before` and `context_after` are optional. Folio mode uses them to send the text around a paragraph, which the editor reads for continuity but does not rewrite.

`effort` is `low`, `medium` (default), or `high`. It maps to `reasoning_effort` on OpenAI and `output_config.effort` on Anthropic. Current models from both providers reject custom sampling temperature, so there is no temperature setting.

On Claude Opus 5.5, Fable 5.1, and Sonnet 5.5, requests opt into server-side fallback (`fallbacks: "default"`), so a safety-classifier decline is retried on Anthropic's recommended model. A request that still ends in a refusal comes back as an error.

Errors stream back as `{"error": "...", "code": "..."}`. Codes include `missing_key`, `invalid_key`, `key_mismatch` (an Anthropic key sent to an OpenAI model, or the reverse), `permission`, `quota`, `rate_limit`, `model_not_found` and `connection`. Messages are written for the user and never echo the key.

### `POST /variations`

Streams four alternative phrasings for a selected replacement inside the rewritten text.

## Frontend Behavior

- Uses `VITE_API_URL` when provided
- Falls back to `http://localhost:8000` in local development
- Falls back to `/api` in production, matching `vercel.json`

## Deployment

`vercel.json` is set up as a two-service deployment:

- `/` -> `frontend`
- `/api` -> `backend/app/main.py`

## Current State

- `frontend/README.md` is still the default Vite template
- `backend/README.md` is currently empty
- This root README is the primary project documentation
