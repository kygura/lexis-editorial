import json
import os
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass

from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
from pydantic import BaseModel

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
- Be language-agnostic: detect and respect the language of the user's text without bias toward English or any other language
- NEVER translate the user's text into another language unless the user explicitly asks for translation
- Preserve the original language, script, orthography, and dialect/register conventions of the source text
- If the input is multilingual or code-switches between languages, preserve that multilingual structure unless the user explicitly asks you to normalize it
- Any rewrite must remain in the same language as the original passage you are rewriting
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

VARIATIONS_SYSTEM_PROMPT = """You are a literary editor generating alternative phrasings for a specific word or phrase inside a rewritten text.

Rules:
- Return ONLY valid JSON, no markdown, no preamble
- Generate exactly 4 alternatives
- Each alternative must fit naturally in the surrounding context
- Vary approach across the four: different imagery, different rhythm, different connotational weight
- Maintain the same part of speech and syntactic role
- All alternatives should be at least as literary as the given replacement
- Be language-agnostic: detect and respect the language of the surrounding text without bias toward English or any other language
- NEVER translate the text into another language unless the user explicitly asks for translation
- Every variation must remain in the same language, script, and multilingual pattern as the surrounding source text

JSON format:
{"variations": ["alternative 1", "alternative 2", "alternative 3", "alternative 4"]}"""



class AnalyzeRequest(BaseModel):
    text: str
    style_directive: str = ""
    provider: str | None = None
    model: str | None = None
    api_key: str | None = None
    temperature: float | None = None


@dataclass(frozen=True)
class ProviderConfig:
    api_key_env: str
    default_model_env: str
    fallback_model: str
    model_aliases: dict[str, list[str]]
    streamer: Callable[[str, str, str, str, float], AsyncIterator[str]]


def build_system_prompt(style_directive: str = "") -> str:
    prompt = BASE_SYSTEM_PROMPT
    if style_directive and style_directive.strip():
        prompt += (
            "\n\nSTYLE DIRECTIVE — the user has requested you target this specific style:\n"
            f"{style_directive.strip()}"
        )
    return prompt


def resolve_default_provider() -> str:
    provider = os.getenv("DEFAULT_PROVIDER", "openai").strip().lower()
    return provider if provider in PROVIDER_REGISTRY else "openai"


def resolve_default_model(provider: str) -> str:
    config = PROVIDER_REGISTRY[provider]
    return os.getenv(config.default_model_env, os.getenv("DEFAULT_MODEL", config.fallback_model))


def resolve_provider_settings(payload: AnalyzeRequest) -> tuple[str, str, str]:
    provider = (payload.provider or resolve_default_provider()).strip().lower()
    if provider not in PROVIDER_REGISTRY:
        raise ValueError(f"Unsupported provider: {provider}")

    config = PROVIDER_REGISTRY[provider]
    api_key = (payload.api_key or "").strip() or os.getenv(config.api_key_env, "").strip()
    if not api_key:
        raise ValueError(f"Missing API key for provider '{provider}'")

    model = (payload.model or "").strip() or resolve_default_model(provider)
    if not model:
        raise ValueError(f"Missing model for provider '{provider}'")

    return provider, model, api_key


def resolve_model_candidates(provider: str, model: str) -> list[str]:
    config = PROVIDER_REGISTRY[provider]
    candidates = config.model_aliases.get(model, [model])
    unique_candidates: list[str] = []
    for candidate in candidates:
        if candidate and candidate not in unique_candidates:
            unique_candidates.append(candidate)
    return unique_candidates


def resolve_temperature(payload: AnalyzeRequest) -> float:
    if payload.temperature is None:
        return 0.7
    return min(2.0, max(0.0, payload.temperature))


def is_model_unavailable_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(
        token in message
        for token in (
            "model",
            "not found",
            "does not exist",
            "invalid model",
            "unsupported",
        )
    )


async def stream_openai_completion(
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
) -> AsyncIterator[str]:
    client = AsyncOpenAI(api_key=api_key)
    stream = await client.chat.completions.create(
        model=model,
        temperature=temperature,
        stream=True,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    async for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta.content:
            yield delta.content


async def stream_anthropic_completion(
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
) -> AsyncIterator[str]:
    client = AsyncAnthropic(api_key=api_key)
    async with client.messages.stream(
        model=model,
        max_tokens=8192,
        temperature=temperature,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    ) as stream:
        async for text in stream.text_stream:
            if text:
                yield text


PROVIDER_REGISTRY: dict[str, ProviderConfig] = {
    "openai": ProviderConfig(
        api_key_env="OPENAI_API_KEY",
        default_model_env="DEFAULT_OPENAI_MODEL",
        fallback_model="gpt-5.4",
        model_aliases={},
        streamer=stream_openai_completion,
    ),
    "anthropic": ProviderConfig(
        api_key_env="ANTHROPIC_API_KEY",
        default_model_env="DEFAULT_ANTHROPIC_MODEL",
        fallback_model="claude-sonnet-4-6",
        model_aliases={
            "claude-opus-4-6": [
                "claude-opus-4-6",
                "claude-opus-4-1-20250805",
            ],
            "claude-opus-4-6-latest": [
                "claude-opus-4-6",
                "claude-opus-4-1-20250805",
            ],
            "claude-opus-4-1": [
                "claude-opus-4-1",
                "claude-opus-4-1-20250805",
            ],
            "claude-sonnet-4-6": [
                "claude-sonnet-4-6",
                "claude-sonnet-4-20250514",
            ],
            "claude-sonnet-4-6-latest": [
                "claude-sonnet-4-6",
                "claude-sonnet-4-20250514",
            ],
            "claude-sonnet-4-5": [
                "claude-sonnet-4-5",
                "claude-sonnet-4-20250514",
            ],
        },
        streamer=stream_anthropic_completion,
    ),
}


def sse_payload(payload: dict[str, str]) -> str:
    return f"data: {json.dumps(payload)}\n\n"


class VariationsRequest(BaseModel):
    rewrite: str
    original: str
    replacement: str
    reason: str = ""
    style_directive: str = ""
    provider: str | None = None
    model: str | None = None
    api_key: str | None = None
    temperature: float | None = None


@app.get("/healthz")
async def healthz():
    return {"status": "ok", "default_provider": resolve_default_provider()}


@app.post("/variations")
async def variations(payload: VariationsRequest):
    rewrite = payload.rewrite.strip()
    if not payload.original or not payload.replacement:
        async def bad_request_stream():
            yield sse_payload({"error": "Missing original or replacement"})

        return StreamingResponse(bad_request_stream(), media_type="text/event-stream")

    user_prompt = (
        f'Full text context:\n"{rewrite}"\n\n'
        f'The word/phrase "{payload.original}" was replaced with "{payload.replacement}".\n'
        f'Editorial reason: {payload.reason}\n\n'
        f'Generate 4 alternative phrasings that could replace "{payload.replacement}" at the same position. '
        f'Return JSON: {{"variations": ["...", "...", "...", "..."]}}'
    )

    async def stream_response():
        try:
            provider, model, api_key = resolve_provider_settings(
                AnalyzeRequest(
                    text=payload.rewrite,
                    style_directive=payload.style_directive,
                    provider=payload.provider,
                    model=payload.model,
                    api_key=payload.api_key,
                    temperature=payload.temperature,
                )
            )
            temperature = resolve_temperature(
                AnalyzeRequest(
                    text="x",
                    temperature=payload.temperature,
                )
            )
            streamer = PROVIDER_REGISTRY[provider].streamer
            candidates = resolve_model_candidates(provider, model)
            last_error: Exception | None = None

            for index, candidate_model in enumerate(candidates):
                try:
                    async for delta in streamer(
                        api_key,
                        candidate_model,
                        VARIATIONS_SYSTEM_PROMPT,
                        user_prompt,
                        temperature,
                    ):
                        yield sse_payload({"text": delta})
                    yield "data: [DONE]\n\n"
                    return
                except Exception as exc:
                    last_error = exc
                    is_last_candidate = index == len(candidates) - 1
                    if is_last_candidate or not is_model_unavailable_error(exc):
                        raise

            if last_error is not None:
                raise last_error
        except Exception as exc:
            yield sse_payload({"error": str(exc)})

    return StreamingResponse(
        stream_response(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/analyze")
async def analyze(payload: AnalyzeRequest):
    text = payload.text.strip()
    if len(text) < 8:
        async def short_text_stream():
            yield sse_payload({"error": "Text too short"})

        return StreamingResponse(short_text_stream(), media_type="text/event-stream")

    system_prompt = build_system_prompt(payload.style_directive)
    user_prompt = f'\n\n"{text}"'

    async def stream_response():
        try:
            provider, model, api_key = resolve_provider_settings(payload)
            temperature = resolve_temperature(payload)
            streamer = PROVIDER_REGISTRY[provider].streamer
            candidates = resolve_model_candidates(provider, model)
            last_error: Exception | None = None

            for index, candidate_model in enumerate(candidates):
                try:
                    async for delta in streamer(
                        api_key,
                        candidate_model,
                        system_prompt,
                        user_prompt,
                        temperature,
                    ):
                        yield sse_payload({"text": delta})
                    yield "data: [DONE]\n\n"
                    return
                except Exception as exc:
                    last_error = exc
                    is_last_candidate = index == len(candidates) - 1
                    if is_last_candidate or not is_model_unavailable_error(exc):
                        raise

            if last_error is not None:
                raise last_error
        except Exception as exc:  # pragma: no cover - surfaced to client
            yield sse_payload({"error": str(exc)})

    return StreamingResponse(
        stream_response(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
