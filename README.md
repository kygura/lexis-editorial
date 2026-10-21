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
|  |- src/App.tsx        # Main single-page application
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

- Available models: OpenAI `gpt-6.1-sol`, `gpt-6-astra`, `gpt-6-luna`; Anthropic `claude-opus-5-5`, `claude-fable-5-1`, `claude-sonnet-5-5`, `claude-haiku-5-5`.
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
  "effort": "medium"
}
```

`effort` is `low`, `medium` (default), or `high`. It maps to `reasoning_effort` on OpenAI and `output_config.effort` on Anthropic. Current models from both providers reject custom sampling temperature, so there is no temperature setting.

On Claude Opus 5.5, Fable 5.1, and Sonnet 5.5, requests opt into server-side fallback (`fallbacks: "default"`), so a safety-classifier decline is retried on Anthropic's recommended model. A request that still ends in a refusal comes back as an error.

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
