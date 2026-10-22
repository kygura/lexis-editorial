import { useCallback, useEffect, useRef, useState } from "react";
import "./App.css";

const LOCAL_BACKEND_URL = "http://localhost:8000"
const API_URL = import.meta.env.VITE_API_URL ||
 (import.meta.env.PROD ? "/api" : LOCAL_BACKEND_URL);
const MIN_LENGTH = 8;
const SIMILARITY_THRESHOLD = 0.85;

const STORAGE_KEYS = {
  theme: "lexis-theme",
  style: "lexis-style-directive",
  provider: "lexis-provider",
  model: "lexis-model",
  effort: "lexis-effort",
  apiKeys: "lexis-api-keys",
  guideSeen: "lexis-guide-seen",
} as const;

const MODEL_OPTIONS = {
  openai: [
    { value: "gpt-6.1-sol", label: "GPT-6.1 Sol", note: "Flagship. Balanced, quick, exacting." },
    { value: "gpt-6-astra", label: "GPT-6 Astra", note: "Deepest reasoning. Slow, meticulous." },
    { value: "gpt-5.6-terra", label: "GPT-5.6 Terra", note: "Even-handed, between Sol and Luna." },
    { value: "gpt-6-luna", label: "GPT-6 Luna", note: "Light and fast for short passages." },
  ],
  anthropic: [
    { value: "claude-opus-5-5", label: "Claude Opus 5.5", note: "Literary range with a fine ear for voice." },
    { value: "claude-fable-5-1", label: "Claude Fable 5.1", note: "Most capable. Long, demanding pieces." },
    { value: "claude-sonnet-5-5", label: "Claude Sonnet 5.5", note: "Fast and capable for everyday drafts." },
    { value: "claude-haiku-5-5", label: "Claude Haiku 5.5", note: "Swift, light-touch line edits." },
  ],
} as const;

const PROVIDER_LABELS = { openai: "OpenAI", anthropic: "Anthropic" } as const;

const EFFORT_OPTIONS = [
  { value: "low", label: "Light", hint: "A quick pass over the line." },
  { value: "medium", label: "Close", hint: "A careful read, sentence by sentence." },
  { value: "high", label: "Deep", hint: "Slow, deliberate, willing to rebuild." },
] as const;

type Effort = (typeof EFFORT_OPTIONS)[number]["value"];
const DEFAULT_EFFORT: Effort = "medium";

const GUIDE_SECTIONS = [
  {
    eyebrow: "I",
    title: "Type the manuscript",
    description: "The left sheet holds your draft. Nothing is sent until you ask.",
    bullets: [
      "Paste or type at least 8 characters.",
      "Press Send to editor, or Ctrl/Cmd + Enter.",
      "The proof sets itself line by line as the editor writes.",
    ],
  },
  {
    eyebrow: "II",
    title: "Choose your editor",
    description: "The model name in the masthead opens the roster.",
    bullets: [
      "Pick from the current OpenAI and Anthropic models.",
      "Model, effort, brief, theme, and keys are saved in this browser.",
    ],
  },
  {
    eyebrow: "III",
    title: "Leave keys at the desk",
    description: "Keys in the masthead stores a per-provider API key locally.",
    bullets: [
      "Paste a key to use it for your requests.",
      "Clear it to fall back to the server's key, if one is set.",
    ],
  },
  {
    eyebrow: "IV",
    title: "Write the brief",
    description: "The editor's brief steers voice, cadence, and constraints.",
    bullets: [
      "Try: tighter, more formal, more lyrical, or a named register.",
      "Effort sets how hard the editor works: Light, Close, or Deep.",
      "Deep takes longer and rebuilds more freely.",
    ],
  },
  {
    eyebrow: "V",
    title: "Read the red pencil",
    description: "Every change is marked in the proof and numbered in the margin notes.",
    bullets: [
      "Click a marked phrase to compare the original and the edit.",
      "Ask for variations to see four other ways to say it.",
      "Click a margin note to restore your original wording.",
    ],
  },
  {
    eyebrow: "VI",
    title: "Revise and keep",
    description: "The proof is yours to work on once it is set.",
    bullets: [
      "Reset sends the manuscript again with current settings.",
      "Edit opens the proof for hand corrections; Save keeps them.",
      "Copy takes the proof as shown, restorations included.",
      "Undo and Redo cover variations, restorations, and saved edits.",
    ],
  },
] as const;

const GUIDE_SHORTCUTS = [
  { keys: "Ctrl/Cmd + Enter", action: "Send the manuscript" },
  { keys: "Ctrl/Cmd + Z", action: "Undo the last proof change" },
  { keys: "Ctrl/Cmd + Shift + Z", action: "Redo" },
  { keys: "Esc", action: "Close panels or the guide" },
] as const;

type ProviderId = keyof typeof MODEL_OPTIONS;
type ProviderKeyMap = Record<ProviderId, string>;

interface RawChange {
  original: string;
  replacement: string;
  start: number;
  end: number;
  reason: string;
}

interface SanitizedChange extends RawChange {
  key: string;
  resolvedStart: number | null;
  resolvedEnd: number | null;
  originalIndex: number;
}

interface HistoryEntry {
  rewrite: string;
  changes: RawChange[];
  toggledChanges: Set<string>;
}

interface Analysis {
  register: string;
  tone: string;
  notes?: string;
}

interface AnalysisResult {
  analysis: Analysis;
  rewrite: string;
  changes: RawChange[];
}

interface LexisError {
  message: string;
  code?: string;
}

// Errors the writer fixes from the Keys panel.
const KEY_ERROR_CODES = new Set(["missing_key", "invalid_key", "key_mismatch", "permission", "quota"]);

class ApiError extends Error {
  code?: string;
  constructor(message: string, code?: string) {
    super(message);
    this.code = code;
  }
}

function toLexisError(caught: unknown, fallback: string): LexisError {
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

interface Segment {
  type: "plain" | "highlight";
  text: string;
  change?: SanitizedChange;
}

function textSimilarity(a: string, b: string): number {
  if (a === b) return 1;
  if (!a.length || !b.length) return 0;
  const longer = a.length >= b.length ? a : b;
  const shorter = a.length >= b.length ? b : a;
  if (longer.length === 0) return 1;
  let matches = 0;
  const windowSize = 2;
  const longerBigrams = new Map<string, number>();
  for (let i = 0; i < longer.length - windowSize + 1; i++) {
    const bigram = longer.substring(i, i + windowSize);
    longerBigrams.set(bigram, (longerBigrams.get(bigram) || 0) + 1);
  }
  for (let i = 0; i < shorter.length - windowSize + 1; i++) {
    const bigram = shorter.substring(i, i + windowSize);
    const count = longerBigrams.get(bigram) || 0;
    if (count > 0) {
      longerBigrams.set(bigram, count - 1);
      matches++;
    }
  }
  return (2.0 * matches) / (longer.length + shorter.length - 2 * (windowSize - 1));
}

function normalizeEffort(value: string | null): Effort {
  return EFFORT_OPTIONS.some((option) => option.value === value) ? (value as Effort) : DEFAULT_EFFORT;
}

function isEditableTarget(target: EventTarget | null) {
  if (!(target instanceof HTMLElement)) return false;
  return !!target.closest("input, textarea, select, [contenteditable='true']") || target.isContentEditable;
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

function sanitizeChanges(rewrite: string, changes: RawChange[]): SanitizedChange[] {
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

function buildSegments(rewrite: string, changes: SanitizedChange[]): Segment[] {
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

function applyChangeToggles(rewrite: string, changes: SanitizedChange[], toggledChanges: Set<string>) {
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
function extractPartialRewrite(raw: string): string {
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

function modelLabel(provider: ProviderId, model: string) {
  return MODEL_OPTIONS[provider].find((option) => option.value === model)?.label ?? model;
}

function defaultModelFor(provider: ProviderId) {
  return MODEL_OPTIONS[provider][0].value;
}

function normalizeProvider(value: string | null): ProviderId {
  return value === "anthropic" ? "anthropic" : "openai";
}

function emptyProviderKeys(): ProviderKeyMap {
  return {
    openai: "",
    anthropic: "",
  };
}

function normalizeProviderKeys(value: unknown): ProviderKeyMap {
  const empty = emptyProviderKeys();
  if (!value || typeof value !== "object") return empty;
  return {
    openai: typeof (value as ProviderKeyMap).openai === "string" ? (value as ProviderKeyMap).openai : "",
    anthropic:
      typeof (value as ProviderKeyMap).anthropic === "string" ? (value as ProviderKeyMap).anthropic : "",
  };
}

function readStorage(key: string) {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

const TODAY = new Date().toLocaleDateString(undefined, { year: "numeric", month: "long", day: "numeric" });

function App() {
  const [input, setInput] = useState("");
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [streaming, setStreaming] = useState(false);
  const [streamPreview, setStreamPreview] = useState("");
  const [activeChangeKey, setActiveChangeKey] = useState<string | null>(null);
  const [modalPos, setModalPos] = useState<{ x: number; y: number; yAbove: number } | null>(null);
  const [error, setError] = useState<LexisError | null>(null);
  const [varError, setVarError] = useState<string | null>(null);
  const [sentPulse, setSentPulse] = useState(false);
  const [dark, setDark] = useState(() => readStorage(STORAGE_KEYS.theme) === "dark");
  const [copied, setCopied] = useState(false);
  const [toggledChanges, setToggledChanges] = useState<Set<string>>(new Set());
  const [styleDirective, setStyleDirective] = useState(() => readStorage(STORAGE_KEYS.style) ?? "");
  const [styleOpen, setStyleOpen] = useState(false);
  const [selectorOpen, setSelectorOpen] = useState(false);
  const [keysOpen, setKeysOpen] = useState(false);
  const [helpOpen, setHelpOpen] = useState(false);
  const [selectedProvider, setSelectedProvider] = useState<ProviderId>(() =>
    normalizeProvider(readStorage(STORAGE_KEYS.provider)),
  );
  const [selectedModel, setSelectedModel] = useState<string>(
    () => readStorage(STORAGE_KEYS.model) || defaultModelFor(normalizeProvider(readStorage(STORAGE_KEYS.provider))),
  );
  const [effort, setEffort] = useState<Effort>(() => normalizeEffort(readStorage(STORAGE_KEYS.effort)));
  const [providerApiKeys, setProviderApiKeys] = useState<ProviderKeyMap>(() => {
    try {
      return normalizeProviderKeys(JSON.parse(readStorage(STORAGE_KEYS.apiKeys) || "null"));
    } catch {
      return emptyProviderKeys();
    }
  });
  const [editMode, setEditMode] = useState(false);
  const [editedText, setEditedText] = useState("");
  const [varLoading, setVarLoading] = useState(false);
  const [varOptions, setVarOptions] = useState<string[]>([]);
  const [varChangeKey, setVarChangeKey] = useState<string | null>(null);
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [redoHistory, setRedoHistory] = useState<HistoryEntry[]>([]);
  const abortRef = useRef<AbortController | null>(null);
  const modalRef = useRef<HTMLDivElement | null>(null);
  const selectorRef = useRef<HTMLDivElement | null>(null);
  const keysRef = useRef<HTMLDivElement | null>(null);
  const styleRef = useRef(styleDirective);
  const providerRef = useRef<ProviderId>(selectedProvider);
  const modelRef = useRef<string>(selectedModel);
  const effortRef = useRef<Effort>(effort);
  const apiKeyRef = useRef("");
  const cachedResultRef = useRef<AnalysisResult | null>(null);
  const cachedInputRef = useRef("");
  const cachedSettingsRef = useRef("");
  const lastCompletedKeyRef = useRef("");
  const inFlightKeyRef = useRef<string | null>(null);
  const requestSequenceRef = useRef(0);
  const varAbortRef = useRef<AbortController | null>(null);

  const sanitizedChanges = result ? sanitizeChanges(result.rewrite, result.changes ?? []) : [];
  const activeChange =
    sanitizedChanges.find((change) => change.key === activeChangeKey) ?? null;
  const segments = result ? buildSegments(result.rewrite, sanitizedChanges) : [];
  const changeNumbers = new Map(sanitizedChanges.map((change, index) => [change.key, index + 1]));

  useEffect(() => {
    if (localStorage.getItem(STORAGE_KEYS.guideSeen) !== "true") {
      setHelpOpen(true);
      localStorage.setItem(STORAGE_KEYS.guideSeen, "true");
    }
  }, []);

  useEffect(() => {
    localStorage.setItem(STORAGE_KEYS.theme, dark ? "dark" : "light");
    document.documentElement.dataset.theme = dark ? "dark" : "light";
  }, [dark]);

  useEffect(() => {
    localStorage.setItem(STORAGE_KEYS.style, styleDirective);
  }, [styleDirective]);

  useEffect(() => {
    localStorage.setItem(STORAGE_KEYS.provider, selectedProvider);
  }, [selectedProvider]);

  useEffect(() => {
    localStorage.setItem(STORAGE_KEYS.model, selectedModel);
  }, [selectedModel]);

  useEffect(() => {
    localStorage.setItem(STORAGE_KEYS.effort, effort);
  }, [effort]);

  useEffect(() => {
    localStorage.setItem(STORAGE_KEYS.apiKeys, JSON.stringify(providerApiKeys));
  }, [providerApiKeys]);

  useEffect(() => {
    const availableModels: string[] = MODEL_OPTIONS[selectedProvider].map((option) => option.value);
    if (!availableModels.includes(selectedModel)) {
      setSelectedModel(defaultModelFor(selectedProvider));
    }
  }, [selectedModel, selectedProvider]);

  useEffect(() => {
    styleRef.current = styleDirective;
  }, [styleDirective]);

  useEffect(() => {
    providerRef.current = selectedProvider;
  }, [selectedProvider]);

  useEffect(() => {
    modelRef.current = selectedModel;
  }, [selectedModel]);

  useEffect(() => {
    effortRef.current = effort;
  }, [effort]);

  useEffect(() => {
    apiKeyRef.current = providerApiKeys[selectedProvider].trim();
  }, [providerApiKeys, selectedProvider]);

  useEffect(() => {
    const handler = (event: MouseEvent) => {
      if (modalRef.current && !modalRef.current.contains(event.target as Node)) {
        setActiveChangeKey(null);
        setModalPos(null);
      }

      if (selectorRef.current && !selectorRef.current.contains(event.target as Node)) {
        setSelectorOpen(false);
      }

      if (keysRef.current && !keysRef.current.contains(event.target as Node)) {
        setKeysOpen(false);
      }
    };

    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  const analyze = useCallback(async (text: string, force = false) => {
    const trimmedText = text.trim();
    const settingsKey = JSON.stringify({
      styleDirective: styleRef.current,
      provider: providerRef.current,
      model: modelRef.current,
      effort: effortRef.current,
      apiKey: apiKeyRef.current,
    });
    const requestKey = JSON.stringify({ text: trimmedText, settingsKey });

    if (!force && inFlightKeyRef.current === requestKey) return;

    if (!force && cachedResultRef.current && cachedSettingsRef.current === settingsKey) {
      const similarity = textSimilarity(trimmedText, cachedInputRef.current);
      if (similarity >= SIMILARITY_THRESHOLD) {
        if (abortRef.current) abortRef.current.abort();
        lastCompletedKeyRef.current = requestKey;
        setStreaming(false);
        setError(null);
        setResult(cachedResultRef.current);
        return;
      }
    }

    if (!force && lastCompletedKeyRef.current === requestKey) {
      return;
    }

    if (abortRef.current) abortRef.current.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    requestSequenceRef.current += 1;
    const requestId = requestSequenceRef.current;
    inFlightKeyRef.current = requestKey;

    setStreaming(true);
    setStreamPreview("");
    setResult(null);
    setError(null);
    setActiveChangeKey(null);
    setModalPos(null);
    setToggledChanges(new Set());
    setHistory([]);
    setRedoHistory([]);
    setEditMode(false);
    setVarOptions([]);
    setVarChangeKey(null);

    try {
      const response = await fetch(API_URL + "/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: controller.signal,
        body: JSON.stringify({
          text: trimmedText,
          style_directive: styleRef.current,
          provider: providerRef.current,
          model: modelRef.current,
          effort: effortRef.current,
          api_key: apiKeyRef.current || undefined,
        }),
      });

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

          try {
            const parsed = JSON.parse(data);
            if (parsed.error) {
              throw new ApiError(parsed.error, parsed.code);
            }
            if (parsed.text) {
              fullText += parsed.text;
            }
          } catch (parseError) {
            if (parseError instanceof ApiError) {
              throw parseError;
            }
          }
        }

        if (requestId === requestSequenceRef.current) {
          setStreamPreview(extractPartialRewrite(fullText));
        }
      }

      if (requestId !== requestSequenceRef.current) return;

      const clean = fullText.replace(/```json|```/g, "").trim();
      const parsed = JSON.parse(clean) as AnalysisResult;
      cachedResultRef.current = parsed;
      cachedInputRef.current = trimmedText;
      cachedSettingsRef.current = settingsKey;
      lastCompletedKeyRef.current = requestKey;
      setResult(parsed);
    } catch (caughtError: unknown) {
      if (caughtError instanceof Error && caughtError.name === "AbortError") return;
      if (requestId !== requestSequenceRef.current) return;
      lastCompletedKeyRef.current = "";
      setError(toLexisError(caughtError, "The rewrite failed. Try again."));
    } finally {
      if (requestId === requestSequenceRef.current) {
        setStreaming(false);
        setStreamPreview("");
      }
      if (inFlightKeyRef.current === requestKey) {
        inFlightKeyRef.current = null;
      }
    }
  }, []);

  const sendNow = useCallback(() => {
    const trimmedInput = input.trim();
    if (trimmedInput.length < MIN_LENGTH) return;

    styleRef.current = styleDirective;
    providerRef.current = selectedProvider;
    modelRef.current = selectedModel;
    effortRef.current = effort;
    apiKeyRef.current = providerApiKeys[selectedProvider].trim();

    // Restart the carriage-return bump even if the last one is still playing.
    setSentPulse(false);
    requestAnimationFrame(() => setSentPulse(true));
    void analyze(trimmedInput, true);
  }, [analyze, effort, input, providerApiKeys, selectedModel, selectedProvider, styleDirective]);

  const snapshotCurrentState = () => {
    if (!result) return null;
    return {
      rewrite: result.rewrite,
      changes: result.changes ?? [],
      toggledChanges: new Set(toggledChanges),
    };
  };

  const pushToHistory = () => {
    const snapshot = snapshotCurrentState();
    if (!snapshot) return;
    setHistory((prev) => [...prev, snapshot]);
    setRedoHistory([]);
  };

  const undo = () => {
    const current = snapshotCurrentState();
    if (!current) return;

    setHistory((prev) => {
      if (prev.length === 0) return prev;
      const entry = prev[prev.length - 1];
      setRedoHistory((redoPrev) => [...redoPrev, current]);
      setResult((r) => (r ? { ...r, rewrite: entry.rewrite, changes: entry.changes } : null));
      setToggledChanges(new Set(entry.toggledChanges));
      setActiveChangeKey(null);
      setModalPos(null);
      setVarOptions([]);
      setVarChangeKey(null);
      return prev.slice(0, -1);
    });
  };

  const redo = () => {
    const current = snapshotCurrentState();
    if (!current) return;

    setRedoHistory((prev) => {
      if (prev.length === 0) return prev;
      const entry = prev[prev.length - 1];
      setHistory((historyPrev) => [...historyPrev, current]);
      setResult((r) => (r ? { ...r, rewrite: entry.rewrite, changes: entry.changes } : null));
      setToggledChanges(new Set(entry.toggledChanges));
      setActiveChangeKey(null);
      setModalPos(null);
      setVarOptions([]);
      setVarChangeKey(null);
      return prev.slice(0, -1);
    });
  };

  useEffect(() => {
    document.body.style.overflow = helpOpen ? "hidden" : "";
    return () => {
      document.body.style.overflow = "";
    };
  }, [helpOpen]);

  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        if (helpOpen || activeChangeKey || selectorOpen || keysOpen || editMode) {
          event.preventDefault();
        }
        setHelpOpen(false);
        setActiveChangeKey(null);
        setModalPos(null);
        setSelectorOpen(false);
        setKeysOpen(false);
        setEditMode(false);
        return;
      }

      if (helpOpen) return;
      if (!(event.ctrlKey || event.metaKey) || event.altKey) return;
      if (isEditableTarget(event.target)) return;

      const key = event.key.toLowerCase();

      if (key === "z" && !event.shiftKey) {
        if (history.length === 0) return;
        event.preventDefault();
        undo();
        return;
      }

      if ((key === "z" && event.shiftKey) || key === "y") {
        if (redoHistory.length === 0) return;
        event.preventDefault();
        redo();
      }
    };

    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [activeChangeKey, editMode, helpOpen, history.length, keysOpen, redo, redoHistory.length, selectorOpen, undo]);

  const enterEditMode = () => {
    if (!result) return;
    const effective = applyChangeToggles(result.rewrite, sanitizedChanges, toggledChanges);
    setEditedText(effective);
    setEditMode(true);
    setActiveChangeKey(null);
    setModalPos(null);
  };

  const saveEdit = () => {
    if (!result) return;
    pushToHistory();
    setResult({ ...result, rewrite: editedText });
    setToggledChanges(new Set());
    setEditMode(false);
    setActiveChangeKey(null);
    setModalPos(null);
  };

  const generateVariations = async (change: SanitizedChange) => {
    if (!result) return;
    setVarLoading(true);
    setVarOptions([]);
    setVarError(null);
    setVarChangeKey(change.key);
    if (varAbortRef.current) varAbortRef.current.abort();
    const controller = new AbortController();
    varAbortRef.current = controller;
    try {
      const response = await fetch(API_URL + "/variations", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: controller.signal,
        body: JSON.stringify({
          rewrite: result.rewrite,
          original: change.original,
          replacement: change.replacement,
          reason: change.reason,
          style_directive: styleRef.current,
          provider: providerRef.current,
          model: modelRef.current,
          effort: effortRef.current,
          api_key: apiKeyRef.current || undefined,
        }),
      });
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
          let parsed: { text?: string; error?: string } | null = null;
          try { parsed = JSON.parse(data); } catch { continue; }
          if (parsed?.error) throw new ApiError(parsed.error, (parsed as { code?: string }).code);
          if (parsed?.text) fullText += parsed.text;
        }
      }
      const clean = fullText.replace(/```json|```/g, "").trim();
      const parsed = JSON.parse(clean);
      setVarOptions(Array.isArray(parsed.variations) ? parsed.variations.slice(0, 4) : []);
    } catch (err) {
      if (err instanceof Error && err.name === "AbortError") return;
      setVarOptions([]);
      setVarError(toLexisError(err, "Couldn't draft variations. Try again.").message);
    } finally {
      setVarLoading(false);
    }
  };

  const applyVariation = (change: SanitizedChange, variation: string) => {
    if (!result || change.resolvedStart === null || change.resolvedEnd === null) return;
    pushToHistory();
    const newRewrite =
      result.rewrite.slice(0, change.resolvedStart) +
      variation +
      result.rewrite.slice(change.resolvedEnd);
    const newChanges = (result.changes ?? []).map((c, i) =>
      i === change.originalIndex
        ? { ...c, replacement: variation, start: change.resolvedStart!, end: change.resolvedStart! + variation.length }
        : c,
    );
    setResult({ ...result, rewrite: newRewrite, changes: newChanges });
    setToggledChanges((prev) => {
      const next = new Set(prev);
      next.delete(change.key);
      return next;
    });
    const newKey = `change-${change.originalIndex}-${change.resolvedStart!}-${change.resolvedStart! + variation.length}`;
    setActiveChangeKey(newKey);
    setVarOptions([]);
    setVarChangeKey(null);
  };

  const toggleChange = (change: SanitizedChange) => {
    if (change.resolvedStart === null || change.resolvedEnd === null) return;
    pushToHistory();
    setToggledChanges((prev) => {
      const next = new Set(prev);
      if (next.has(change.key)) {
        next.delete(change.key);
      } else {
        next.add(change.key);
      }
      return next;
    });
  };

  const copyRewrite = () => {
    if (!result?.rewrite) return;
    const text = applyChangeToggles(result.rewrite, sanitizedChanges, toggledChanges);
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleHighlightClick = (change: SanitizedChange, event: React.MouseEvent) => {
    const rect = (event.currentTarget as HTMLElement).getBoundingClientRect();
    if (activeChangeKey === change.key) {
      setActiveChangeKey(null);
      setModalPos(null);
      return;
    }

    setActiveChangeKey(change.key);
    setModalPos({
      x: rect.left + rect.width / 2,
      y: rect.bottom + 10,
      yAbove: rect.top - 10,
    });
  };

  const hasContent = input.trim().length >= MIN_LENGTH;
  const wordCount = input.trim().split(/\s+/).filter(Boolean).length;
  const proofWordCount = result ? result.rewrite.trim().split(/\s+/).filter(Boolean).length : 0;
  const canSend = hasContent && !streaming;
  const handleModelSelection = (provider: ProviderId, model: string) => {
    setSelectedProvider(provider);
    setSelectedModel(model);
    setSelectorOpen(false);
  };
  const handleProviderApiKeyChange = (provider: ProviderId, value: string) => {
    setProviderApiKeys((prev) => ({
      ...prev,
      [provider]: value,
    }));
  };

  const activeIsToggled = activeChange ? toggledChanges.has(activeChange.key) : false;
  const activeCanToggle =
    !!activeChange && activeChange.resolvedStart !== null && activeChange.resolvedEnd !== null;

  return (
    <div className="lx-app">
      <header className="lx-masthead">
        <div className="lx-masthead-top">
          <span className="lx-dateline">{TODAY}</span>
          <h1 className="lx-wordmark">Lexis</h1>
          <span className="lx-dateline lx-dateline-right">An editorial companion</span>
        </div>

        <nav className="lx-toolbar" aria-label="Settings">
          <div ref={selectorRef} className="lx-popover-anchor">
            <button
              className="lx-chip"
              aria-expanded={selectorOpen}
              onClick={() => setSelectorOpen(!selectorOpen)}
            >
              <span className="lx-chip-label">Editor</span>
              <span className="lx-chip-value">{modelLabel(selectedProvider, selectedModel)}</span>
              <span className="lx-caret" aria-hidden>▾</span>
            </button>

            {selectorOpen ? (
              <div className="lx-popover lx-popover-left lx-roster" role="listbox" aria-label="Choose model">
                {(Object.keys(MODEL_OPTIONS) as ProviderId[]).map((provider) => (
                  <div key={provider} className="lx-roster-group">
                    <div className="lx-eyebrow">{PROVIDER_LABELS[provider]}</div>
                    {MODEL_OPTIONS[provider].map((option) => {
                      const isSelected = provider === selectedProvider && option.value === selectedModel;
                      return (
                        <button
                          key={option.value}
                          role="option"
                          aria-selected={isSelected}
                          className={"lx-roster-item" + (isSelected ? " is-selected" : "")}
                          onClick={() => handleModelSelection(provider, option.value)}
                        >
                          <span className="lx-roster-name">{option.label}</span>
                          <span className="lx-roster-note">{option.note}</span>
                          <span className="lx-roster-id">{option.value}</span>
                        </button>
                      );
                    })}
                  </div>
                ))}
              </div>
            ) : null}
          </div>

          <div className="lx-effort" role="radiogroup" aria-label="Effort">
            <span className="lx-chip-label">Effort</span>
            {EFFORT_OPTIONS.map((option) => (
              <button
                key={option.value}
                role="radio"
                aria-checked={effort === option.value}
                title={option.hint}
                className={"lx-effort-key" + (effort === option.value ? " is-on" : "")}
                onClick={() => setEffort(option.value)}
              >
                {option.label}
              </button>
            ))}
          </div>

          <span className="lx-toolbar-spacer" />

          <span className={"lx-status" + (streaming ? " is-busy" : "")} aria-live="polite">
            {streaming ? "Setting type" : result ? `${sanitizedChanges.length} mark${sanitizedChanges.length !== 1 ? "s" : ""}` : ""}
          </span>

          <button
            className="lx-link"
            onClick={() => {
              setHelpOpen(true);
              setSelectorOpen(false);
              setKeysOpen(false);
              localStorage.setItem(STORAGE_KEYS.guideSeen, "true");
            }}
          >
            Guide
          </button>

          <div ref={keysRef} className="lx-popover-anchor">
            <button className="lx-link" aria-expanded={keysOpen} onClick={() => setKeysOpen(!keysOpen)}>
              Keys
            </button>

            {keysOpen ? (
              <div className="lx-popover lx-popover-right lx-keys">
                <div className="lx-eyebrow">Provider keys</div>
                {(Object.keys(MODEL_OPTIONS) as ProviderId[]).map((provider) => (
                  <label key={provider} className="lx-field">
                    <span className="lx-field-head">
                      <span className={provider === selectedProvider ? "is-current" : undefined}>
                        {PROVIDER_LABELS[provider]}
                      </span>
                      <button
                        type="button"
                        className="lx-link lx-link-small"
                        onClick={() => handleProviderApiKeyChange(provider, "")}
                        disabled={!providerApiKeys[provider]}
                      >
                        Clear
                      </button>
                    </span>
                    <input
                      type="password"
                      autoFocus={provider === selectedProvider}
                      className="lx-input"
                      value={providerApiKeys[provider]}
                      onChange={(event) => handleProviderApiKeyChange(provider, event.target.value)}
                      placeholder={`Use the server's ${PROVIDER_LABELS[provider]} key`}
                    />
                  </label>
                ))}
              </div>
            ) : null}
          </div>

          <button className="lx-link" onClick={() => setDark(!dark)}>
            {dark ? "Paper" : "Carbon"}
          </button>
        </nav>
      </header>

      <main className="lx-desk">
        <section
          className={"lx-sheet lx-manuscript" + (sentPulse ? " is-sent" : "")}
          aria-label="Manuscript"
          onAnimationEnd={(event) => {
            if (event.target === event.currentTarget) setSentPulse(false);
          }}
        >
          <div className="lx-sheet-head">
            <span className="lx-eyebrow">Manuscript</span>
            <span className="lx-meta">{input.length > 0 ? `${wordCount} words` : "Draft"}</span>
          </div>

          <textarea
            autoFocus
            className="lx-typewriter"
            value={input}
            spellCheck
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={(event) => {
              if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
                event.preventDefault();
                sendNow();
              }
            }}
            placeholder={"Begin typing…"}
          />

          <div className="lx-brief">
            <button
              className={"lx-brief-toggle" + (styleDirective ? " has-brief" : "")}
              aria-expanded={styleOpen}
              onClick={() => setStyleOpen(!styleOpen)}
            >
              <span className={"lx-disclosure" + (styleOpen ? " is-open" : "")} aria-hidden>▸</span>
              Editor's brief
              {styleDirective && !styleOpen ? <span className="lx-brief-peek">“{styleDirective}”</span> : null}
            </button>

            {styleOpen ? (
              <div className="lx-brief-body">
                <textarea
                  className="lx-input lx-brief-input"
                  value={styleDirective}
                  onChange={(event) => setStyleDirective(event.target.value)}
                  placeholder={"e.g. Write like Cormac McCarthy: sparse punctuation, biblical cadence, visceral imagery…"}
                />
                {styleDirective ? (
                  <button className="lx-link lx-link-small" onClick={() => setStyleDirective("")}>
                    Clear brief
                  </button>
                ) : null}
              </div>
            ) : null}
          </div>

          <div className="lx-sheet-foot">
            <span className="lx-meta">Ctrl/Cmd + Enter</span>
            <button className="lx-key" onClick={sendNow} disabled={!canSend}>
              {streaming ? "With the editor…" : "Send to editor"}
              <span className="lx-key-glyph" aria-hidden>⏎</span>
            </button>
          </div>
        </section>

        <section className="lx-sheet lx-proof" aria-label="Proof" aria-busy={streaming}>
          <div className="lx-sheet-head">
            <span className="lx-eyebrow">Proof</span>
            <div className="lx-actions">
              {history.length > 0 && !streaming ? (
                <button className="lx-link" onClick={undo}>Undo</button>
              ) : null}
              {redoHistory.length > 0 && !streaming ? (
                <button className="lx-link" onClick={redo}>Redo</button>
              ) : null}
              {result && !streaming && !editMode ? (
                <>
                  <button className="lx-link" onClick={sendNow}>Reset</button>
                  <button className="lx-link" onClick={enterEditMode}>Edit</button>
                </>
              ) : null}
              {editMode ? (
                <>
                  <button className="lx-link" onClick={() => setEditMode(false)}>Cancel</button>
                  <button className="lx-link is-pencil" onClick={saveEdit}>Save</button>
                </>
              ) : null}
              {result && !editMode ? (
                <button className={"lx-link" + (copied ? " is-done" : "")} onClick={copyRewrite}>
                  {copied ? "Copied" : "Copy"}
                </button>
              ) : null}
            </div>
          </div>

          {result?.analysis && !editMode ? (
            <div className="lx-colophon">
              {result.analysis.register ? <span>Register · {result.analysis.register}</span> : null}
              {result.analysis.tone ? <span>Tone · {result.analysis.tone}</span> : null}
              <span>{proofWordCount} words</span>
            </div>
          ) : null}

          {!result && !streaming && !error ? (
            <p className="lx-proof-text lx-placeholder">
              {hasContent
                ? "Send the manuscript when you're ready. The proof will be set here, with every edit marked in red pencil."
                : "The proof will be set here."}
            </p>
          ) : null}

          {streaming && !result ? (
            <p className="lx-proof-text lx-setting">
              {streamPreview}
              <span className="lx-cursor" aria-hidden />
            </p>
          ) : null}

          {result && editMode ? (
            <textarea
              autoFocus
              className="lx-proof-text lx-proof-edit"
              value={editedText}
              onChange={(e) => setEditedText(e.target.value)}
            />
          ) : result ? (
            <p className="lx-proof-text">
              {segments.map((segment, index) => {
                if (segment.type === "plain") {
                  return <span key={`plain-${index}`}>{segment.text}</span>;
                }

                const change = segment.change!;
                const isActive = activeChangeKey === change.key;
                const isToggled = toggledChanges.has(change.key);
                return (
                  <span
                    key={change.key}
                    role="button"
                    tabIndex={0}
                    data-n={changeNumbers.get(change.key)}
                    className={"lx-mark" + (isToggled ? " is-restored" : "") + (isActive ? " is-active" : "")}
                    onClick={(event) => handleHighlightClick(change, event)}
                    onKeyDown={(event) => {
                      if (event.key === "Enter" || event.key === " ") {
                        event.preventDefault();
                        handleHighlightClick(change, event as unknown as React.MouseEvent);
                      }
                    }}
                  >
                    {isToggled ? change.original : segment.text}
                  </span>
                );
              })}
            </p>
          ) : null}

          {error ? (
            <div className="lx-error" role="alert">
              <p>{error.message}</p>
              {error.code && KEY_ERROR_CODES.has(error.code) ? (
                <button
                  className="lx-link"
                  onClick={() => {
                    setKeysOpen(true);
                    setSelectorOpen(false);
                  }}
                >
                  Open keys
                </button>
              ) : null}
            </div>
          ) : null}

          {sanitizedChanges.length > 0 && !editMode ? (
            <aside className="lx-notes" aria-label="Margin notes">
              <div className="lx-eyebrow">Margin notes</div>
              <ol>
                {sanitizedChanges.map((change, index) => {
                  const isToggled = toggledChanges.has(change.key);
                  const canToggle = change.resolvedStart !== null && change.resolvedEnd !== null;
                  return (
                    <li key={change.key}>
                      <button
                        className={"lx-note" + (isToggled ? " is-restored" : "")}
                        disabled={!canToggle}
                        title={canToggle ? (isToggled ? "Use the edit" : "Restore the original") : "Not found in proof"}
                        onClick={() => toggleChange(change)}
                      >
                        <span className="lx-note-n">{index + 1}</span>
                        <span className="lx-note-body">
                          <span className="lx-note-swap">
                            <s>{change.original}</s>
                            <span aria-hidden> → </span>
                            <em>{change.replacement}</em>
                          </span>
                          {change.reason ? <span className="lx-note-reason">{change.reason}</span> : null}
                        </span>
                        <span className="lx-note-state">{isToggled ? "stet" : ""}</span>
                      </button>
                    </li>
                  );
                })}
              </ol>
            </aside>
          ) : null}

          {result?.analysis?.notes && !editMode ? (
            <p className="lx-editor-note">
              <span className="lx-eyebrow">Editor's note</span>
              {result.analysis.notes}
            </p>
          ) : null}
        </section>
      </main>

      {helpOpen ? (
        <div className="lx-scrim" onClick={() => setHelpOpen(false)}>
          <div
            className="lx-guide"
            role="dialog"
            aria-modal="true"
            aria-label="How to use Lexis"
            onClick={(event) => event.stopPropagation()}
          >
            <button className="lx-close" aria-label="Close guide" onClick={() => setHelpOpen(false)}>
              ×
            </button>

            <div className="lx-eyebrow">How to use Lexis</div>
            <h2 className="lx-guide-title">Type the draft. Read the proof. Keep what sings.</h2>
            <p className="lx-guide-lede">
              You write on the left. An editor returns a typeset proof on the right, every change marked in red
              pencil and annotated in the margin. Restore, swap, or hand-correct any of it.
            </p>

            <div className="lx-shortcuts">
              {GUIDE_SHORTCUTS.map((shortcut) => (
                <div key={shortcut.keys} className="lx-shortcut">
                  <kbd>{shortcut.keys}</kbd>
                  <span>{shortcut.action}</span>
                </div>
              ))}
            </div>

            <div className="lx-guide-grid">
              {GUIDE_SECTIONS.map((section) => (
                <section key={section.title} className="lx-guide-section">
                  <div className="lx-guide-numeral">{section.eyebrow}</div>
                  <h3>{section.title}</h3>
                  <p>{section.description}</p>
                  <ul>
                    {section.bullets.map((bullet) => (
                      <li key={bullet}>{bullet}</li>
                    ))}
                  </ul>
                </section>
              ))}
            </div>
          </div>
        </div>
      ) : null}

      {activeChange && modalPos ? (
        <div
          ref={modalRef}
          className="lx-card"
          role="dialog"
          aria-label="Edit details"
          style={{
            left: Math.max(16, Math.min(modalPos.x - 210, window.innerWidth - 436)),
            top: (() => {
              const approxHeight = 440;
              const fitsBelow = modalPos.y + approxHeight < window.innerHeight - 16;
              return fitsBelow ? modalPos.y : Math.max(16, modalPos.yAbove - approxHeight);
            })(),
          }}
        >
          <button
            className="lx-close"
            aria-label="Close"
            onClick={() => {
              setActiveChangeKey(null);
              setModalPos(null);
            }}
          >
            ×
          </button>

          <div className="lx-eyebrow">Mark {changeNumbers.get(activeChange.key)}</div>

          <div className="lx-compare">
            <button
              className={"lx-compare-side" + (activeIsToggled ? " is-chosen" : "")}
              disabled={!activeCanToggle}
              onClick={() => {
                if (!activeIsToggled && activeCanToggle) toggleChange(activeChange);
              }}
            >
              <span className="lx-eyebrow">Original</span>
              <span className="lx-compare-text">{activeChange.original}</span>
            </button>
            <button
              className={"lx-compare-side" + (!activeIsToggled ? " is-chosen" : "")}
              disabled={!activeCanToggle}
              onClick={() => {
                if (activeIsToggled && activeCanToggle) toggleChange(activeChange);
              }}
            >
              <span className="lx-eyebrow">Edit</span>
              <span className="lx-compare-text">{activeChange.replacement}</span>
            </button>
          </div>

          {activeChange.reason ? <p className="lx-card-reason">{activeChange.reason}</p> : null}

          <div className="lx-card-variations">
            <div className="lx-card-variations-head">
              <span className="lx-eyebrow">Variations</span>
              <button
                className="lx-link"
                onClick={() => generateVariations(activeChange)}
                disabled={varLoading}
              >
                {varLoading && varChangeKey === activeChange.key ? "Drafting…" : varOptions.length > 0 && varChangeKey === activeChange.key ? "Again" : "Draft four"}
              </button>
            </div>

            {varLoading && varChangeKey === activeChange.key ? (
              <p className="lx-meta lx-setting-small">
                Drafting alternatives<span className="lx-cursor" aria-hidden />
              </p>
            ) : varError && varChangeKey === activeChange.key ? (
              <p className="lx-card-error">{varError}</p>
            ) : varOptions.length > 0 && varChangeKey === activeChange.key ? (
              <div className="lx-variations">
                {varOptions.map((v, i) => (
                  <button key={i} className="lx-variation" onClick={() => applyVariation(activeChange, v)}>
                    <span className="lx-variation-n">{String.fromCharCode(97 + i)}</span>
                    {v}
                  </button>
                ))}
              </div>
            ) : null}
          </div>
        </div>
      ) : null}
    </div>
  );
}

export default App;
