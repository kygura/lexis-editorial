export interface RawChange {
  original: string;
  replacement: string;
  start: number;
  end: number;
  reason: string;
}

export interface SanitizedChange extends RawChange {
  key: string;
  resolvedStart: number | null;
  resolvedEnd: number | null;
  originalIndex: number;
}

export interface Analysis {
  register: string;
  tone: string;
  notes?: string;
}

export interface AnalysisResult {
  analysis: Analysis;
  rewrite: string;
  changes: RawChange[];
}

export interface LexisError {
  message: string;
  code?: string;
}

// Errors the writer fixes from the Keys panel.
export const KEY_ERROR_CODES = new Set(["missing_key", "invalid_key", "key_mismatch", "permission", "quota"]);

export class ApiError extends Error {
  code?: string;
  constructor(message: string, code?: string) {
    super(message);
    this.code = code;
  }
}

export function toLexisError(caught: unknown, fallback: string): LexisError {
  if (caught instanceof ApiError) return { message: caught.message, code: caught.code };
  // fetch() rejects with a TypeError when the Lexis server can't be reached.
  if (caught instanceof TypeError) {
    return { message: "Couldn't reach the Lexis server. Check that the backend is running.", code: "server_unreachable" };
  }
  if (caught instanceof SyntaxError) {
    return { message: "The editor's reply came back garbled. Send again.", code: "bad_response" };
  }
  return { message: fallback };
}

export interface Segment {
  type: "plain" | "highlight";
  text: string;
  change?: SanitizedChange;
}

function rangesOverlap(aStart: number, aEnd: number, bStart: number, bEnd: number) {
  return aStart < bEnd && bStart < aEnd;
}

function findOccurrenceCandidates(text: string, needle: string) {
  const indices: number[] = [];
  if (!needle) return indices;
  let fromIndex = 0;
  while (fromIndex <= text.length) {
    const index = text.indexOf(needle, fromIndex);
    if (index === -1) break;
    indices.push(index);
    fromIndex = index + Math.max(1, needle.length);
  }
  return indices;
}

export function sanitizeChanges(rewrite: string, changes: RawChange[]): SanitizedChange[] {
  if (!changes.length) return [];

  const usedRanges: Array<{ start: number; end: number }> = [];
  const sanitized: SanitizedChange[] = new Array(changes.length);
  const indices = changes
    .map((change, index) => ({ change, index }))
    .sort((a, b) => a.change.start - b.change.start || a.index - b.index);

  for (const { change, index } of indices) {
    let resolvedStart: number | null = null;
    let resolvedEnd: number | null = null;
    const hintedStart = Number.isFinite(change.start) ? Math.max(0, change.start) : 0;
    const hintedEnd = Number.isFinite(change.end) ? Math.max(hintedStart, change.end) : hintedStart;
    const hintedMatches =
      change.replacement &&
      hintedEnd <= rewrite.length &&
      rewrite.slice(hintedStart, hintedEnd) === change.replacement &&
      !usedRanges.some((range) => rangesOverlap(hintedStart, hintedEnd, range.start, range.end));

    if (hintedMatches) {
      resolvedStart = hintedStart;
      resolvedEnd = hintedEnd;
    } else if (change.replacement) {
      const replacementLength = change.replacement.length;
      const candidates = findOccurrenceCandidates(rewrite, change.replacement)
        .filter((candidateStart) => {
          const candidateEnd = candidateStart + replacementLength;
          return !usedRanges.some((range) =>
            rangesOverlap(candidateStart, candidateEnd, range.start, range.end),
          );
        })
        .map((candidateStart) => ({
          start: candidateStart,
          end: candidateStart + replacementLength,
          distance: Math.abs(candidateStart - hintedStart),
        }))
        .sort((a, b) => a.distance - b.distance || a.start - b.start);

      if (candidates.length > 0) {
        resolvedStart = candidates[0].start;
        resolvedEnd = candidates[0].end;
      }
    }

    if (resolvedStart !== null && resolvedEnd !== null) {
      usedRanges.push({ start: resolvedStart, end: resolvedEnd });
    }

    sanitized[index] = {
      ...change,
      key: `change-${index}-${resolvedStart ?? "na"}-${resolvedEnd ?? "na"}`,
      resolvedStart,
      resolvedEnd,
      originalIndex: index,
    };
  }

  return sanitized;
}

export function buildSegments(rewrite: string, changes: SanitizedChange[]): Segment[] {
  const resolvedChanges = changes
    .filter(
      (change): change is SanitizedChange & { resolvedStart: number; resolvedEnd: number } =>
        change.resolvedStart !== null && change.resolvedEnd !== null,
    )
    .sort((a, b) => a.resolvedStart - b.resolvedStart);

  if (!resolvedChanges.length) return [{ type: "plain", text: rewrite }];

  const segments: Segment[] = [];
  let cursor = 0;

  for (const change of resolvedChanges) {
    if (change.resolvedStart < cursor) continue;
    if (change.resolvedStart > cursor) {
      segments.push({ type: "plain", text: rewrite.slice(cursor, change.resolvedStart) });
    }
    segments.push({
      type: "highlight",
      text: rewrite.slice(change.resolvedStart, change.resolvedEnd),
      change,
    });
    cursor = change.resolvedEnd;
  }

  if (cursor < rewrite.length) {
    segments.push({ type: "plain", text: rewrite.slice(cursor) });
  }

  return segments;
}

export function applyChangeToggles(rewrite: string, changes: SanitizedChange[], toggledChanges: Set<string>) {
  let nextRewrite = rewrite;
  const reversibleChanges = changes
    .filter(
      (change): change is SanitizedChange & { resolvedStart: number; resolvedEnd: number } =>
        toggledChanges.has(change.key) &&
        change.resolvedStart !== null &&
        change.resolvedEnd !== null,
    )
    .sort((a, b) => b.resolvedStart - a.resolvedStart);

  for (const change of reversibleChanges) {
    nextRewrite =
      nextRewrite.slice(0, change.resolvedStart) +
      change.original +
      nextRewrite.slice(change.resolvedEnd);
  }

  return nextRewrite;
}

// Pulls the "rewrite" string out of a JSON response that is still streaming,
// so the proof can be set as the model writes it.
export function extractPartialRewrite(raw: string): string {
  const match = raw.match(/"rewrite"\s*:\s*"/);
  if (!match || match.index === undefined) return "";
  const escapes: Record<string, string> = { n: "\n", t: "\t", r: "", b: "", f: "", '"': '"', "\\": "\\", "/": "/" };
  let out = "";
  let i = match.index + match[0].length;
  while (i < raw.length) {
    const ch = raw[i];
    if (ch === '"') break;
    if (ch === "\\") {
      const next = raw[i + 1];
      if (next === undefined) break;
      if (next === "u") {
        const hex = raw.slice(i + 2, i + 6);
        if (hex.length < 4) break;
        out += String.fromCharCode(Number.parseInt(hex, 16));
        i += 6;
        continue;
      }
      out += escapes[next] ?? next;
      i += 2;
      continue;
    }
    out += ch;
    i++;
  }
  return out;
}

// Reads a Lexis SSE stream, calling onText with the accumulated model output
// after each network chunk. Resolves with the full output.
export async function readEditorStream(response: Response, onText?: (fullText: string) => void) {
  if (!response.ok || !response.body) {
    throw new ApiError(`The Lexis server returned an error (${response.status}). Try again.`, "server");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let lineBuffer = "";
  let fullText = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    lineBuffer += decoder.decode(value, { stream: true });
    const lines = lineBuffer.split("\n");
    lineBuffer = lines.pop() || "";

    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      const data = line.slice(6).trim();
      if (data === "[DONE]") continue;
      let parsed: { text?: string; error?: string; code?: string } | null = null;
      try {
        parsed = JSON.parse(data);
      } catch {
        continue;
      }
      if (parsed?.error) throw new ApiError(parsed.error, parsed.code);
      if (parsed?.text) fullText += parsed.text;
    }

    onText?.(fullText);
  }

  return fullText;
}

export function parseEditorJson<T>(fullText: string): T {
  return JSON.parse(fullText.replace(/```json|```/g, "").trim()) as T;
}
