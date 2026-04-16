from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
import json

load_dotenv()

app = FastAPI()

# Disable CORS. Do not remove this for full-stack development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

BASE_SYSTEM_PROMPT = """You are an editorial companion — a prose stylist who rewrites text for narrative aesthetic. You don't just swap rare words; you reshape sentences, elevate rhythm, and conjure richer imagery while preserving the author's voice.

Rules:
- Return ONLY valid JSON, no markdown, no preamble
- Produce a full rewrite of the text in "rewrite" — this is the primary output
- Prioritize narrative beauty: cadence, imagery, emotional resonance, connotational depth
- You may restructure entire sentences or clauses if it serves the prose
- Lean toward: evocative phrasing, literary vocabulary, precise sensory language, stronger connotational weight
- Never flatten complexity or simplify unless the original is genuinely clumsy
- Maintain the register and intent of the original
- In "changes", list each meaningful substitution you made
- CRITICAL: "start" and "end" are character indices in the REWRITE text (0-indexed), pointing to where each replacement appears in the rewrite
- Double-check that rewrite[start:end] exactly matches the "replacement" string

JSON format:
{
  "analysis": {
    "register": "formal|literary|casual|technical|journalistic",
    "tone": "brief description",
    "notes": "optional stylistic observation about the original"
  },
  "rewrite": "the full rewritten text here",
  "changes": [
    {
      "original": "exact word or phrase from the ORIGINAL text",
      "replacement": "the corresponding word or phrase in the REWRITE",
      "start": character_index_start_in_rewrite,
      "end": character_index_end_in_rewrite,
      "reason": "why this change elevates the prose"
    }
  ]
}"""


def build_system_prompt(style_directive: str = "") -> str:
    prompt = BASE_SYSTEM_PROMPT
    if style_directive and style_directive.strip():
        prompt += f"\n\nSTYLE DIRECTIVE — the user has requested you target this specific style:\n{style_directive.strip()}"
    return prompt

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.post("/api/analyze")
async def analyze(request: Request):
    body = await request.json()
    text = body.get("text", "")
    style_directive = body.get("style_directive", "")

    if not text or len(text.strip()) < 8:
        return {"error": "Text too short"}

    system_prompt = build_system_prompt(style_directive)

    async def stream_response():
        try:
            stream = await client.chat.completions.create(
                model="gpt-5.4",
                max_completion_tokens=100000,
                temperature=0.7,
                stream=True,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f'Analyze this text:\n\n"{text}"'},
                ],
            )

            async for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield f"data: {json.dumps({'text': delta.content})}\n\n"

            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        stream_response(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
