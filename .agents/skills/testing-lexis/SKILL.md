# Testing Lexis

## Deployed URLs
- **Frontend:** https://vocab-variation-app-deebqliu.devinapps.com
- **Backend:** https://app-snlabxpf.fly.dev/

## Local Startup
```bash
# Backend
cd lexis-backend && poetry run fastapi dev app/main.py
# Frontend (separate terminal)
cd lexis-frontend && npm run dev
```

## Key Features to Test
1. **Rewrite generation** — Type 8+ chars, wait for 900ms debounce, verify REWRITING animation and result in right panel
2. **Inline highlighting** — Click highlighted words in the rewrite to open floating modal with original/replacement/reason
3. **Changelog toggle** — Click changelog pills to swap between original and edited words (clean swap, no strikethrough)
4. **Dark/light mode** — Toggle via top-right button, persists in localStorage
5. **Copy button** — Copies rewrite text respecting any toggled changes
6. **Input caching** — Small edits (85%+ similarity) reuse cached result. Regenerate button forces fresh API call
7. **Style directive** — Collapsible panel at bottom of left panel. Describe a target style, verify output reflects it. Changing style invalidates cache. Persists in localStorage

## Testing the Style Directive
- Use a distinctive style like "Write like a pirate" to verify the AI follows the directive
- Clear the directive and verify the output reverts to standard editorial prose
- Style changes should trigger a new API call even when input text hasn't changed

## Type Checking
```bash
cd lexis-frontend && npx tsc --noEmit
```

## API Key
The backend requires `OPENAI_API_KEY` set in `lexis-backend/.env`. Model: gpt-5.4.
