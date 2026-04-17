import { useCallback, useEffect, useRef, useState } from "react";
import "./App.css";

const API_URL = import.meta.env.VITE_API_URL || (import.meta.env.PROD ? "/api" : "http://localhost:8000");
const DEBOUNCE_MS = 900;
const MIN_LENGTH = 8;
const SIMILARITY_THRESHOLD = 0.85;

const STORAGE_KEYS = {
  theme: "lexis-theme",
  style: "lexis-style-directive",
  provider: "lexis-provider",
  model: "lexis-model",
  temperature: "lexis-temperature",
  apiKeys: "lexis-api-keys",
} as const;

const MODEL_OPTIONS = {
  openai: [
    { value: "gpt-5.4", label: "GPT-5.4", defaultTemperature: 0.7 },
    { value: "gpt-4.1", label: "GPT-4.1", defaultTemperature: 0.7 },
    { value: "gpt-4o", label: "GPT-4o", defaultTemperature: 0.7 },
  ],
  anthropic: [
    { value: "claude-sonnet-4-6", label: "Claude Sonnet 4.6", defaultTemperature: 0.7 },
    { value: "claude-opus-4-6", label: "Claude Opus 4.6", defaultTemperature: 0.7 },
    { value: "claude-opus-4-1-20250805", label: "Claude Opus 4.1", defaultTemperature: 0.7 },
    { value: "claude-sonnet-4-20250514", label: "Claude Sonnet 4", defaultTemperature: 0.7 },
    { value: "claude-3-7-sonnet-latest", label: "Claude 3.7 Sonnet", defaultTemperature: 0.7 },
  ],
} as const;

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

function useDebounce<T>(value: T, delay: number) {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const t = setTimeout(() => setDebounced(value), delay);
    return () => clearTimeout(t);
  }, [value, delay]);
  return debounced;
}

function clampTemperature(value: number) {
  if (Number.isNaN(value)) return 0.7;
  return Math.min(2, Math.max(0, value));
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

function allModelOptions() {
  return (Object.entries(MODEL_OPTIONS) as Array<[ProviderId, (typeof MODEL_OPTIONS)[ProviderId]]>).flatMap(
    ([provider, options]) => options.map((option) => ({ ...option, provider })),
  );
}

function defaultModelFor(provider: ProviderId) {
  return MODEL_OPTIONS[provider][0].value;
}

function normalizeProvider(value: string | null): ProviderId {
  return value === "anthropic" ? "anthropic" : "openai";
}

function defaultTemperatureForModel(provider: ProviderId, model: string) {
  const option = MODEL_OPTIONS[provider].find((candidate) => candidate.value === model);
  return clampTemperature(option?.defaultTemperature ?? 0.7);
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

function App() {
  const [input, setInput] = useState("");
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [streaming, setStreaming] = useState(false);
  const [activeChangeKey, setActiveChangeKey] = useState<string | null>(null);
  const [modalPos, setModalPos] = useState<{ x: number; y: number } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dark, setDark] = useState(true);
  const [copied, setCopied] = useState(false);
  const [toggledChanges, setToggledChanges] = useState<Set<string>>(new Set());
  const [styleDirective, setStyleDirective] = useState("");
  const [styleOpen, setStyleOpen] = useState(false);
  const [selectorOpen, setSelectorOpen] = useState(false);
  const [keysOpen, setKeysOpen] = useState(false);
  const [selectedProvider, setSelectedProvider] = useState<ProviderId>("openai");
  const [selectedModel, setSelectedModel] = useState(defaultModelFor("openai"));
  const [temperature, setTemperature] = useState(defaultTemperatureForModel("openai", defaultModelFor("openai")));
  const [providerApiKeys, setProviderApiKeys] = useState<ProviderKeyMap>(emptyProviderKeys());
  const abortRef = useRef<AbortController | null>(null);
  const modalRef = useRef<HTMLDivElement | null>(null);
  const selectorRef = useRef<HTMLDivElement | null>(null);
  const keysRef = useRef<HTMLDivElement | null>(null);
  const styleRef = useRef("");
  const providerRef = useRef<ProviderId>("openai");
  const modelRef = useRef(defaultModelFor("openai"));
  const temperatureRef = useRef(defaultTemperatureForModel("openai", defaultModelFor("openai")));
  const apiKeyRef = useRef("");
  const cachedResultRef = useRef<AnalysisResult | null>(null);
  const cachedInputRef = useRef("");
  const cachedSettingsRef = useRef("");
  const lastCompletedKeyRef = useRef("");
  const inFlightKeyRef = useRef<string | null>(null);
  const requestSequenceRef = useRef(0);
  const debouncedInput = useDebounce(input, DEBOUNCE_MS);
  const debouncedStyle = useDebounce(styleDirective, DEBOUNCE_MS);
  const debouncedTemperature = useDebounce(temperature, DEBOUNCE_MS);
  const debouncedProviderApiKeys = useDebounce(providerApiKeys, DEBOUNCE_MS);

  const sanitizedChanges = result ? sanitizeChanges(result.rewrite, result.changes ?? []) : [];
  const activeChange =
    sanitizedChanges.find((change) => change.key === activeChangeKey) ?? null;
  const segments = result ? buildSegments(result.rewrite, sanitizedChanges) : [];

  useEffect(() => {
    const savedTheme = localStorage.getItem(STORAGE_KEYS.theme);
    const savedStyle = localStorage.getItem(STORAGE_KEYS.style);
    const savedProvider = normalizeProvider(localStorage.getItem(STORAGE_KEYS.provider));
    const savedModel = localStorage.getItem(STORAGE_KEYS.model);
    const savedTemperature = Number.parseFloat(localStorage.getItem(STORAGE_KEYS.temperature) || "");
    const savedApiKeys = localStorage.getItem(STORAGE_KEYS.apiKeys);

    if (savedTheme) setDark(savedTheme === "dark");
    if (savedStyle) setStyleDirective(savedStyle);
    setSelectedProvider(savedProvider);
    const nextModel = savedModel || defaultModelFor(savedProvider);
    setSelectedModel(nextModel);
    setTemperature(
      Number.isFinite(savedTemperature)
        ? clampTemperature(savedTemperature)
        : defaultTemperatureForModel(savedProvider, nextModel),
    );
    if (savedApiKeys) {
      try {
        setProviderApiKeys(normalizeProviderKeys(JSON.parse(savedApiKeys)));
      } catch {
        setProviderApiKeys(emptyProviderKeys());
      }
    }
  }, []);

  useEffect(() => {
    localStorage.setItem(STORAGE_KEYS.theme, dark ? "dark" : "light");
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
    localStorage.setItem(STORAGE_KEYS.temperature, temperature.toString());
  }, [temperature]);

  useEffect(() => {
    localStorage.setItem(STORAGE_KEYS.apiKeys, JSON.stringify(providerApiKeys));
  }, [providerApiKeys]);

  useEffect(() => {
    const availableModels = MODEL_OPTIONS[selectedProvider].map((option) => option.value);
    if (!availableModels.includes(selectedModel)) {
      const fallbackModel = defaultModelFor(selectedProvider);
      setSelectedModel(fallbackModel);
      setTemperature(defaultTemperatureForModel(selectedProvider, fallbackModel));
    }
  }, [selectedModel, selectedProvider]);

  useEffect(() => {
    styleRef.current = debouncedStyle;
  }, [debouncedStyle]);

  useEffect(() => {
    providerRef.current = selectedProvider;
  }, [selectedProvider]);

  useEffect(() => {
    modelRef.current = selectedModel;
  }, [selectedModel]);

  useEffect(() => {
    temperatureRef.current = clampTemperature(debouncedTemperature);
  }, [debouncedTemperature]);

  useEffect(() => {
    apiKeyRef.current = debouncedProviderApiKeys[selectedProvider].trim();
  }, [debouncedProviderApiKeys, selectedProvider]);

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
      temperature: temperatureRef.current,
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
    setResult(null);
    setError(null);
    setActiveChangeKey(null);
    setModalPos(null);
    setToggledChanges(new Set());

    try {
      const response = await fetch(API_URL + "/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: controller.signal,
        body: JSON.stringify({
          text: trimmedText,
          style_directive: styleRef.current,
          provider: providerRef.current,
          model: modelRef.current,
          temperature: temperatureRef.current,
          api_key: apiKeyRef.current || undefined,
        }),
      });

      if (!response.ok) {
        throw new Error("Analysis failed. Try again.");
      }

      if (!response.body) {
        throw new Error("Analysis failed. Try again.");
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
              throw new Error(parsed.error);
            }
            if (parsed.text) {
              fullText += parsed.text;
            }
          } catch (parseError) {
            if (parseError instanceof Error) {
              throw parseError;
            }
          }
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
      setError(caughtError instanceof Error ? caughtError.message : "Analysis failed. Try again.");
    } finally {
      if (requestId === requestSequenceRef.current) {
        setStreaming(false);
      }
      if (inFlightKeyRef.current === requestKey) {
        inFlightKeyRef.current = null;
      }
    }
  }, []);

  useEffect(() => {
    const trimmedInput = debouncedInput.trim();
    if (trimmedInput.length >= MIN_LENGTH) {
      analyze(trimmedInput);
      return;
    }

    if (abortRef.current) abortRef.current.abort();
    lastCompletedKeyRef.current = "";
    inFlightKeyRef.current = null;
    setResult(null);
    setStreaming(false);
    setActiveChangeKey(null);
    setModalPos(null);
    setError(null);
  }, [analyze, debouncedInput]);

  useEffect(() => {
    const trimmedInput = debouncedInput.trim();
    if (trimmedInput.length < MIN_LENGTH) return;
    analyze(trimmedInput);
  }, [analyze, debouncedProviderApiKeys, debouncedStyle, debouncedTemperature, selectedModel, selectedProvider, debouncedInput]);

  const toggleChange = (change: SanitizedChange) => {
    if (change.resolvedStart === null || change.resolvedEnd === null) return;

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
    const rect = (event.target as HTMLElement).getBoundingClientRect();
    if (activeChangeKey === change.key) {
      setActiveChangeKey(null);
      setModalPos(null);
      return;
    }

    setActiveChangeKey(change.key);
    setModalPos({
      x: rect.left + rect.width / 2,
      y: rect.bottom + 8,
    });
  };

  const hasContent = input.trim().length >= MIN_LENGTH;
  const wordCount = input.trim().split(/\s+/).filter(Boolean).length;
  const allModels = allModelOptions();
  const handleModelSelection = (provider: ProviderId, model: string) => {
    setSelectedProvider(provider);
    setSelectedModel(model);
    setTemperature(defaultTemperatureForModel(provider, model));
    setSelectorOpen(false);
  };
  const handleProviderApiKeyChange = (provider: ProviderId, value: string) => {
    setProviderApiKeys((prev) => ({
      ...prev,
      [provider]: value,
    }));
  };

  const d = dark;
  const accent = "#c8945a";
  const theme = {
    bg: d ? "#0e0c0a" : "#f5f2ec",
    bgR: d ? "#110f0c" : "#f0ece4",
    border: d ? "#1e1a14" : "#d8d0c4",
    text: d ? "#e8ddd0" : "#2a2218",
    textMuted: d ? "#5a4e42" : "#a09080",
    textFaint: d ? "#382e24" : "#c8bfb4",
    label: d ? "#4a3e32" : "#c0b0a0",
    unchanged: d ? "#9a8878" : "#4a3a2a",
    accentText: d ? "#c8a87a" : "#7a5a2a",
    hoverBg: d ? "#1e1a14" : "#e8e0d4",
    toggleBg: d ? "#1a1610" : "#ede8e0",
    toggleBorder: d ? "#2a2418" : "#d0c8bc",
    modalBg: d ? "#131008" : "#faf7f1",
    modalBorder: d ? "#2a2418" : "#ddd5c4",
    pillBg: d ? "#1a1610" : "#f2ede3",
    pillBorder: d ? "#252018" : "#e2d8cc",
  };

  const topBorder = "1px solid " + theme.border;
  const panelBorder = "1px solid " + theme.border;
  const toggleBtnBorder = "1px solid " + theme.toggleBorder;
  const copyBtnBorder = "1px solid " + theme.border;
  const modalOuterBorder = "1px solid " + theme.modalBorder;
  const accentBorderBottom = "1.5px solid " + accent;
  const mutedBorderBottom = "1.5px solid " + theme.textMuted;
  const placeholderColor = d ? "#382e24" : "#c8bfb4";
  const scrollThumbColor = d ? "#2a2418" : "#d0c8bc";
  const sliderTrackColor = d ? "#241f17" : "#d8d0c4";
  const sliderThumbOutline = d ? "#2f271b" : "#d3c9bb";

  const cssText = [
    "@keyframes pulse { 0%,100%{opacity:0.4} 50%{opacity:1} }",
    "@keyframes fadeIn { from{opacity:0} to{opacity:1} }",
    "@keyframes slideUp { from{opacity:0;transform:translateY(10px)} to{opacity:1;transform:translateY(0)} }",
    "textarea::placeholder, input::placeholder { font-style: italic; color: " + placeholderColor + "; }",
    "select { appearance: none; }",
    ".lexis-temp-slider { appearance: none; -webkit-appearance: none; width: 100%; height: 34px; background: transparent; margin: 0; cursor: pointer; }",
    ".lexis-temp-slider:focus { outline: none; }",
    ".lexis-temp-slider::-webkit-slider-runnable-track { height: 2px; background: " + sliderTrackColor + "; border-radius: 999px; }",
    ".lexis-temp-slider::-webkit-slider-thumb { -webkit-appearance: none; appearance: none; width: 14px; height: 14px; border-radius: 50%; margin-top: -6px; background: " + accent + "; border: 1px solid " + sliderThumbOutline + "; box-shadow: 0 0 0 2px " + theme.bgR + "; transition: transform 0.14s ease, box-shadow 0.14s ease; }",
    ".lexis-temp-slider:hover::-webkit-slider-thumb { transform: scale(1.05); }",
    ".lexis-temp-slider:focus-visible::-webkit-slider-thumb { box-shadow: 0 0 0 2px " + theme.bgR + ", 0 0 0 4px " + accent + "55; }",
    ".lexis-temp-slider::-moz-range-track { height: 2px; background: " + sliderTrackColor + "; border: none; border-radius: 999px; }",
    ".lexis-temp-slider::-moz-range-thumb { width: 14px; height: 14px; border-radius: 50%; background: " + accent + "; border: 1px solid " + sliderThumbOutline + "; box-shadow: 0 0 0 2px " + theme.bgR + "; transition: transform 0.14s ease, box-shadow 0.14s ease; }",
    ".lexis-temp-slider:hover::-moz-range-thumb { transform: scale(1.05); }",
    ".lexis-temp-slider:focus-visible::-moz-range-thumb { box-shadow: 0 0 0 2px " + theme.bgR + ", 0 0 0 4px " + accent + "55; }",
    "* { box-sizing: border-box; }",
    "::-webkit-scrollbar { width: 3px; }",
    "::-webkit-scrollbar-thumb { background: " + scrollThumbColor + "; }",
    "@media (max-width: 960px) { .lexis-panels { grid-template-columns: 1fr !important; min-height: auto !important; } .lexis-left-panel { border-right: none !important; border-bottom: " + panelBorder + " !important; } .lexis-legend { position: static !important; margin-top: 20px !important; } }",
  ].join("\n");

  return (
    <div
      style={{
        minHeight: "100vh",
        background: theme.bg,
        fontFamily: "Palatino, 'Palatino Linotype', serif",
        display: "flex",
        flexDirection: "column",
        transition: "background 0.25s",
      }}
    >
      <div
        style={{
          padding: "20px 40px",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: 20,
          borderBottom: topBorder,
        }}
      >
        <div style={{ display: "flex", alignItems: "baseline", gap: 10, flexWrap: "wrap" }}>
          <span
            style={{
              fontSize: 11,
              letterSpacing: "0.35em",
              textTransform: "uppercase",
              color: theme.textMuted,
            }}
          >
            Lexis
          </span>
          <span
            style={{
              width: 28,
              height: 1,
              background: theme.border,
              display: "inline-block",
              verticalAlign: "middle",
            }}
          />
          <span
            style={{
              fontSize: 11,
              color: theme.textFaint,
              letterSpacing: "0.12em",
            }}
          >
            editorial companion
          </span>
          <div ref={selectorRef} style={{ position: "relative" }}>
            <button
              onClick={() => setSelectorOpen(!selectorOpen)}
              style={{
                background: "none",
                border: "none",
                color: theme.textMuted,
                fontSize: 10,
                letterSpacing: "0.12em",
                textTransform: "uppercase",
                cursor: "pointer",
                padding: 0,
                display: "flex",
                alignItems: "center",
                gap: 8,
                fontFamily: "inherit",
              }}
              onMouseEnter={(event) => {
                event.currentTarget.style.color = theme.text;
              }}
              onMouseLeave={(event) => {
                event.currentTarget.style.color = theme.textMuted;
              }}
            >
              <span>{selectedProvider}</span>
              <span style={{ color: theme.border }}>/</span>
              <span>{selectedModel}</span>
            </button>

            {selectorOpen ? (
              <div
                style={{
                  position: "absolute",
                  top: "calc(100% + 14px)",
                  left: 0,
                  width: 320,
                  maxWidth: "calc(100vw - 64px)",
                  background: theme.modalBg,
                  border: modalOuterBorder,
                  boxShadow: d ? "0 24px 60px rgba(0,0,0,0.45)" : "0 16px 48px rgba(0,0,0,0.12)",
                  padding: "16px 16px 14px",
                  zIndex: 40,
                  animation: "slideUp 0.18s ease",
                }}
              >
                <div
                  style={{
                    fontSize: 9,
                    letterSpacing: "0.24em",
                    textTransform: "uppercase",
                    color: theme.label,
                    marginBottom: 12,
                  }}
                >
                  Choose model
                </div>
                <div style={{ display: "grid", gap: 8 }}>
                  {allModels.map((option) => {
                    const isSelected =
                      option.provider === selectedProvider && option.value === selectedModel;
                    return (
                      <button
                        key={`${option.provider}-${option.value}`}
                        onClick={() => handleModelSelection(option.provider, option.value)}
                        style={{
                          background: isSelected ? theme.hoverBg : "transparent",
                          border: "1px solid " + (isSelected ? accent : theme.modalBorder),
                          padding: "12px 14px",
                          textAlign: "left",
                          cursor: "pointer",
                          fontFamily: "inherit",
                          color: theme.text,
                          transition: "border-color 0.15s, background 0.15s, color 0.15s",
                        }}
                      >
                        <div
                          style={{
                            display: "flex",
                            justifyContent: "space-between",
                            alignItems: "baseline",
                            gap: 12,
                          }}
                        >
                          <span
                            style={{
                              fontSize: 18,
                              fontStyle: "italic",
                              color: isSelected ? theme.accentText : theme.text,
                            }}
                          >
                            {option.label}
                          </span>
                          <span
                            style={{
                              fontSize: 9,
                              letterSpacing: "0.18em",
                              textTransform: "uppercase",
                              color: isSelected ? theme.accentText : theme.textMuted,
                              whiteSpace: "nowrap",
                            }}
                          >
                            {option.provider}
                          </span>
                        </div>
                      </button>
                    );
                  })}
                </div>
              </div>
            ) : null}
          </div>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 20, flexWrap: "wrap" }}>
          <div ref={keysRef} style={{ position: "relative" }}>
            <button
              onClick={() => setKeysOpen(!keysOpen)}
              style={{
                background: "none",
                border: "none",
                color: theme.textMuted,
                fontSize: 10,
                letterSpacing: "0.18em",
                textTransform: "uppercase",
                cursor: "pointer",
                padding: 0,
                fontFamily: "inherit",
              }}
              onMouseEnter={(event) => {
                event.currentTarget.style.color = theme.text;
              }}
              onMouseLeave={(event) => {
                event.currentTarget.style.color = theme.textMuted;
              }}
            >
              Keys
            </button>

            {keysOpen ? (
              <div
                style={{
                  position: "absolute",
                  top: "calc(100% + 14px)",
                  right: 0,
                  width: 360,
                  maxWidth: "calc(100vw - 64px)",
                  background: theme.modalBg,
                  border: modalOuterBorder,
                  boxShadow: d ? "0 24px 60px rgba(0,0,0,0.45)" : "0 16px 48px rgba(0,0,0,0.12)",
                  padding: "16px 16px 14px",
                  zIndex: 40,
                  animation: "slideUp 0.18s ease",
                }}
              >
                <div
                  style={{
                    fontSize: 9,
                    letterSpacing: "0.24em",
                    textTransform: "uppercase",
                    color: theme.label,
                    marginBottom: 12,
                  }}
                >
                  Provider keys
                </div>
                <div style={{ display: "grid", gap: 12 }}>
                  {(Object.keys(MODEL_OPTIONS) as ProviderId[]).map((provider) => (
                    <label key={provider} style={{ display: "grid", gap: 6 }}>
                      <div
                        style={{
                          display: "flex",
                          alignItems: "baseline",
                          justifyContent: "space-between",
                          gap: 12,
                        }}
                      >
                        <span
                          style={{
                            fontSize: 10,
                            letterSpacing: "0.18em",
                            textTransform: "uppercase",
                            color: provider === selectedProvider ? theme.accentText : theme.label,
                          }}
                        >
                          {provider}
                        </span>
                        <button
                          type="button"
                          onClick={() => handleProviderApiKeyChange(provider, "")}
                          style={{
                            background: "none",
                            border: "none",
                            color: theme.textMuted,
                            fontSize: 9,
                            letterSpacing: "0.15em",
                            textTransform: "uppercase",
                            cursor: "pointer",
                            fontFamily: "inherit",
                            padding: 0,
                            opacity: providerApiKeys[provider] ? 1 : 0.5,
                          }}
                          disabled={!providerApiKeys[provider]}
                        >
                          Clear
                        </button>
                      </div>
                      <input
                        type="password"
                        value={providerApiKeys[provider]}
                        onChange={(event) => handleProviderApiKeyChange(provider, event.target.value)}
                        placeholder={`Use saved ${provider} server key`}
                        style={{
                          width: "100%",
                          background: theme.toggleBg,
                          border: "1px solid " + (provider === selectedProvider ? accent : theme.toggleBorder),
                          outline: "none",
                          fontFamily: "inherit",
                          fontSize: 13,
                          color: theme.text,
                          caretColor: accent,
                          padding: "10px 12px",
                        }}
                        onFocus={(event) => {
                          event.currentTarget.style.borderColor = accent;
                        }}
                        onBlur={(event) => {
                          event.currentTarget.style.borderColor =
                            provider === selectedProvider ? accent : theme.toggleBorder;
                        }}
                      />
                    </label>
                  ))}
                </div>
              </div>
            ) : null}
          </div>

          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 8,
              opacity: hasContent || streaming ? 1 : 0,
              transition: "opacity 0.3s",
            }}
          >
            {streaming ? (
              <>
                {[0, 1, 2].map((i) => (
                  <div
                    key={i}
                    style={{
                      width: 5,
                      height: 5,
                      borderRadius: "50%",
                      background: accent,
                      animation: "pulse 1.2s ease-in-out infinite",
                      animationDelay: i * 180 + "ms",
                    }}
                  />
                ))}
                <span
                  style={{
                    fontSize: 10,
                    letterSpacing: "0.2em",
                    color: theme.textMuted,
                    textTransform: "uppercase",
                    marginLeft: 4,
                  }}
                >
                  rewriting
                </span>
              </>
            ) : result ? (
              <>
                <div
                  style={{
                    width: 6,
                    height: 6,
                    borderRadius: "50%",
                    background: "#7aaa7a",
                    boxShadow: "0 0 8px #7aaa7a80",
                  }}
                />
                <span
                  style={{
                    fontSize: 10,
                    letterSpacing: "0.2em",
                    color: theme.textMuted,
                    textTransform: "uppercase",
                  }}
                >
                  {sanitizedChanges.length} change{sanitizedChanges.length !== 1 ? "s" : ""}
                </span>
              </>
            ) : null}
          </div>

          <button
            onClick={() => setDark(!dark)}
            style={{
              background: theme.toggleBg,
              border: toggleBtnBorder,
              color: theme.textMuted,
              padding: "5px 14px",
              fontSize: 10,
              letterSpacing: "0.2em",
              textTransform: "uppercase",
              cursor: "pointer",
              fontFamily: "inherit",
              transition: "all 0.2s",
            }}
            onMouseEnter={(event) => {
              event.currentTarget.style.color = theme.text;
            }}
            onMouseLeave={(event) => {
              event.currentTarget.style.color = theme.textMuted;
            }}
          >
            {dark ? "Light" : "Dark"}
          </button>
        </div>
      </div>

      <div
        className="lexis-panels"
        style={{
          flex: 1,
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          minHeight: "calc(100vh - 65px)",
        }}
      >
        <div
          className="lexis-left-panel"
          style={{
            padding: 48,
            borderRight: panelBorder,
            display: "flex",
            flexDirection: "column",
            position: "relative",
          }}
        >
          <div
            style={{
              fontSize: 10,
              letterSpacing: "0.3em",
              textTransform: "uppercase",
              color: theme.label,
              marginBottom: 24,
            }}
          >
            Your text
          </div>
          <textarea
            autoFocus
            value={input}
            onChange={(event) => setInput(event.target.value)}
            placeholder={"Begin writing…"}
            style={{
              flex: 1,
              background: "transparent",
              border: "none",
              outline: "none",
              resize: "none",
              fontFamily: "Palatino, serif",
              fontSize: 19,
              lineHeight: 1.85,
              color: theme.text,
              caretColor: accent,
              minHeight: 300,
            }}
          />

          <div
            style={{
              borderTop: panelBorder,
              marginTop: 16,
              paddingTop: 12,
              display: "grid",
              gap: 12,
            }}
          >
            <button
              onClick={() => setStyleOpen(!styleOpen)}
              style={{
                background: "none",
                border: "none",
                color: styleDirective ? accent : theme.textMuted,
                fontSize: 10,
                letterSpacing: "0.2em",
                textTransform: "uppercase",
                cursor: "pointer",
                fontFamily: "inherit",
                padding: 0,
                display: "flex",
                alignItems: "center",
                gap: 6,
              }}
              onMouseEnter={(event) => {
                event.currentTarget.style.color = theme.text;
              }}
              onMouseLeave={(event) => {
                event.currentTarget.style.color = styleDirective ? accent : theme.textMuted;
              }}
            >
              <span
                style={{
                  display: "inline-block",
                  transform: styleOpen ? "rotate(90deg)" : "rotate(0deg)",
                  transition: "transform 0.15s",
                  fontSize: 8,
                }}
              >
                &#9654;
              </span>
              Style directive{styleDirective ? " •" : ""}
            </button>

            {styleOpen ? (
              <div style={{ display: "grid", gap: 12, animation: "fadeIn 0.2s" }}>
                <textarea
                  value={styleDirective}
                  onChange={(event) => setStyleDirective(event.target.value)}
                  placeholder={"e.g. Write in the style of Cormac McCarthy — sparse punctuation, biblical cadence, visceral imagery…"}
                  style={{
                    width: "100%",
                    background: theme.toggleBg,
                    border: toggleBtnBorder,
                    outline: "none",
                    resize: "vertical",
                    fontFamily: "Palatino, serif",
                    fontSize: 13,
                    lineHeight: 1.7,
                    color: theme.text,
                    caretColor: accent,
                    padding: "10px 12px",
                    minHeight: 60,
                    maxHeight: 140,
                    transition: "border-color 0.15s",
                  }}
                  onFocus={(event) => {
                    event.currentTarget.style.borderColor = accent;
                  }}
                  onBlur={(event) => {
                    event.currentTarget.style.borderColor = theme.toggleBorder;
                  }}
                />

                <label style={{ display: "grid", gap: 6 }}>
                  <span
                    style={{
                      fontSize: 10,
                      letterSpacing: "0.18em",
                      textTransform: "uppercase",
                      color: theme.label,
                    }}
                  >
                    Temperature
                  </span>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 88px", gap: 10 }}>
                    <input
                      className="lexis-temp-slider"
                      type="range"
                      min={0}
                      max={2}
                      step={0.05}
                      value={temperature}
                      onChange={(event) => {
                        setTemperature(clampTemperature(Number.parseFloat(event.target.value)));
                      }}
                      style={{ accentColor: accent }}
                    />
                    <input
                      type="number"
                      min={0}
                      max={2}
                      step={0.05}
                      value={temperature}
                      onChange={(event) => {
                        setTemperature(clampTemperature(Number.parseFloat(event.target.value)));
                      }}
                      style={{
                        background: theme.toggleBg,
                        border: toggleBtnBorder,
                        color: theme.text,
                        padding: "10px 12px",
                        fontFamily: "inherit",
                        fontSize: 13,
                        outline: "none",
                      }}
                    />
                  </div>
                </label>

                <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
                  {styleDirective ? (
                    <button
                      onClick={() => setStyleDirective("")}
                      style={{
                        background: "none",
                        border: "none",
                        color: theme.textMuted,
                        fontSize: 9,
                        letterSpacing: "0.15em",
                        textTransform: "uppercase",
                        cursor: "pointer",
                        fontFamily: "inherit",
                        padding: "0",
                      }}
                      onMouseEnter={(event) => {
                        event.currentTarget.style.color = theme.text;
                      }}
                      onMouseLeave={(event) => {
                        event.currentTarget.style.color = theme.textMuted;
                      }}
                    >
                      Clear style
                    </button>
                  ) : null}

                </div>
              </div>
            ) : null}
          </div>

          <div
            style={{
              fontSize: 10,
              color: theme.textFaint,
              letterSpacing: "0.15em",
              marginTop: 12,
            }}
          >
            {input.length > 0 ? wordCount + " words" : ""}
          </div>
        </div>

        <div
          style={{
            padding: 48,
            display: "flex",
            flexDirection: "column",
            position: "relative",
            background: theme.bgR,
            transition: "background 0.25s",
            overflowY: "auto",
          }}
        >
          <div
            style={{
              fontSize: 10,
              letterSpacing: "0.3em",
              textTransform: "uppercase",
              color: theme.label,
              marginBottom: 24,
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              gap: 16,
              flexWrap: "wrap",
            }}
          >
            <span>Alternate</span>
            <div style={{ display: "flex", alignItems: "center", gap: 14, flexWrap: "wrap" }}>
              {result && !streaming ? (
                <span
                  style={{
                    fontSize: 9,
                    color: theme.textFaint,
                    letterSpacing: "0.1em",
                    fontStyle: "italic",
                  }}
                >
                  click highlighted words
                </span>
              ) : null}

              {result && !streaming ? (
                <button
                  onClick={() => analyze(input.trim(), true)}
                  style={{
                    background: "none",
                    border: copyBtnBorder,
                    color: theme.textMuted,
                    padding: "4px 14px",
                    fontSize: 9,
                    letterSpacing: "0.2em",
                    textTransform: "uppercase",
                    cursor: "pointer",
                    fontFamily: "inherit",
                    transition: "all 0.15s",
                  }}
                  onMouseEnter={(event) => {
                    event.currentTarget.style.borderColor = theme.textMuted;
                    event.currentTarget.style.color = theme.text;
                  }}
                  onMouseLeave={(event) => {
                    event.currentTarget.style.borderColor = theme.border;
                    event.currentTarget.style.color = theme.textMuted;
                  }}
                >
                  Regenerate
                </button>
              ) : null}

              {result ? (
                <button
                  onClick={copyRewrite}
                  style={{
                    background: "none",
                    border: copyBtnBorder,
                    color: copied ? "#7aaa7a" : theme.textMuted,
                    padding: "4px 14px",
                    fontSize: 9,
                    letterSpacing: "0.2em",
                    textTransform: "uppercase",
                    cursor: "pointer",
                    fontFamily: "inherit",
                    transition: "all 0.15s",
                  }}
                  onMouseEnter={(event) => {
                    event.currentTarget.style.borderColor = theme.textMuted;
                    event.currentTarget.style.color = theme.text;
                  }}
                  onMouseLeave={(event) => {
                    event.currentTarget.style.borderColor = theme.border;
                    event.currentTarget.style.color = copied ? "#7aaa7a" : theme.textMuted;
                  }}
                >
                  {copied ? "Copied" : "Copy"}
                </button>
              ) : null}
            </div>
          </div>

          {!hasContent && !streaming ? (
            <div
              style={{
                fontSize: 19,
                lineHeight: 1.85,
                color: theme.textFaint,
                fontStyle: "italic",
              }}
            >
              A rewritten version will appear here...
            </div>
          ) : null}

          {hasContent && streaming && !result ? (
            <div
              style={{
                fontSize: 19,
                lineHeight: 1.85,
                color: theme.textFaint,
                fontStyle: "italic",
              }}
            >
              <span style={{ animation: "pulse 1.2s ease-in-out infinite" }}>Rewriting...</span>
            </div>
          ) : null}

          {result ? (
            <div
              style={{
                fontSize: 19,
                lineHeight: 1.85,
                color: theme.text,
                fontFamily: "Palatino, serif",
                whiteSpace: "pre-wrap",
                wordBreak: "break-word",
                marginBottom: 32,
              }}
            >
              {segments.map((segment, index) => {
                if (segment.type === "plain") {
                  return (
                    <span key={`plain-${index}`} style={{ color: theme.unchanged }}>
                      {segment.text}
                    </span>
                  );
                }

                const isActive = activeChangeKey === segment.change?.key;
                const isToggled = toggledChanges.has(segment.change!.key);
                return (
                  <span
                    key={segment.change!.key}
                    onClick={(event) => handleHighlightClick(segment.change!, event)}
                    style={{
                      color: isToggled ? theme.unchanged : theme.accentText,
                      borderBottom: isToggled ? mutedBorderBottom : accentBorderBottom,
                      paddingBottom: 1,
                      cursor: "pointer",
                      padding: "0 1px",
                      transition: "background 0.1s, color 0.15s",
                      background: isActive ? theme.hoverBg : "transparent",
                      textDecoration: "none",
                      fontStyle: "italic",
                    }}
                    onMouseEnter={(event) => {
                      event.currentTarget.style.background = theme.hoverBg;
                    }}
                    onMouseLeave={(event) => {
                      event.currentTarget.style.background = isActive ? theme.hoverBg : "transparent";
                    }}
                  >
                    {isToggled ? segment.change!.original : segment.text}
                  </span>
                );
              })}
            </div>
          ) : null}

          {sanitizedChanges.length > 0 ? (
            <div style={{ marginTop: "auto", paddingTop: 24, borderTop: topBorder }}>
              <div
                style={{
                  fontSize: 10,
                  letterSpacing: "0.3em",
                  textTransform: "uppercase",
                  color: theme.label,
                  marginBottom: 14,
                }}
              >
                Changelog
              </div>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 7 }}>
                {sanitizedChanges.map((change) => {
                  const isToggled = toggledChanges.has(change.key);
                  const pillBorderVal =
                    "1px solid " + (isToggled ? theme.textMuted : theme.pillBorder);
                  const canToggle = change.resolvedStart !== null && change.resolvedEnd !== null;

                  return (
                    <div
                      key={change.key}
                      onClick={() => {
                        if (canToggle) toggleChange(change);
                      }}
                      style={{
                        background: isToggled ? (d ? "#1a1a14" : "#e8e4dc") : theme.pillBg,
                        border: pillBorderVal,
                        padding: "8px 14px",
                        cursor: canToggle ? "pointer" : "default",
                        display: "flex",
                        alignItems: "center",
                        gap: 8,
                        transition: "border-color 0.12s, background 0.12s",
                        fontSize: 14,
                        fontFamily: "Palatino, serif",
                        opacity: canToggle ? 1 : 0.6,
                      }}
                      onMouseEnter={(event) => {
                        if (canToggle) event.currentTarget.style.borderColor = accent;
                      }}
                      onMouseLeave={(event) => {
                        event.currentTarget.style.borderColor = isToggled
                          ? theme.textMuted
                          : theme.pillBorder;
                      }}
                    >
                      <span
                        style={{
                          fontStyle: "italic",
                          color: isToggled ? theme.accentText : theme.textMuted,
                        }}
                      >
                        {change.original}
                      </span>
                      <span style={{ color: theme.textFaint, fontSize: 12 }}>
                        {canToggle ? (isToggled ? "←" : "→") : "•"}
                      </span>
                      <span
                        style={{
                          fontStyle: "italic",
                          color: isToggled ? theme.textMuted : theme.accentText,
                        }}
                      >
                        {change.replacement}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          ) : null}

          {result?.analysis?.notes ? (
            <div
              style={{
                fontSize: 12,
                fontStyle: "italic",
                color: theme.textMuted,
                marginTop: 18,
                lineHeight: 1.6,
              }}
            >
              <span style={{ color: accent, marginRight: 4 }}>&#10022;</span>
              {result.analysis.notes}
            </div>
          ) : null}

          {result ? (
            <div
              className="lexis-legend"
              style={{
                position: "absolute",
                bottom: 32,
                left: 48,
                display: "flex",
                alignItems: "center",
                gap: 16,
                fontSize: 10,
                color: theme.textFaint,
                letterSpacing: "0.12em",
              }}
            >
              <span style={{ display: "flex", alignItems: "center", gap: 5 }}>
                <span style={{ width: 16, height: 1.5, background: accent, display: "inline-block" }} />
                changed
              </span>
              <span style={{ color: theme.border }}>&middot;</span>
              <span>unchanged</span>
            </div>
          ) : null}

          {error ? (
            <div style={{ color: "#aa5050", fontSize: 13, marginTop: 16, fontStyle: "italic" }}>
              {error}
            </div>
          ) : null}
        </div>
      </div>

      {activeChange && modalPos ? (
        <div
          ref={modalRef}
          style={{
            position: "fixed",
            zIndex: 50,
            width: 420,
            maxWidth: "92vw",
            background: theme.modalBg,
            border: modalOuterBorder,
            boxShadow: d ? "0 24px 60px rgba(0,0,0,0.6)" : "0 16px 48px rgba(0,0,0,0.12)",
            left: Math.max(16, Math.min(modalPos.x - 210, window.innerWidth - 436)),
            top: modalPos.y,
            animation: "slideUp 0.18s ease",
          }}
        >
          <div style={{ padding: "24px 28px 20px", position: "relative" }}>
            <button
              onClick={() => {
                setActiveChangeKey(null);
                setModalPos(null);
              }}
              style={{
                position: "absolute",
                top: 12,
                right: 14,
                background: "none",
                border: "none",
                cursor: "pointer",
                color: d ? "#5a4e3a" : "#b0a090",
                fontSize: 18,
                lineHeight: 1,
                padding: 4,
                transition: "color 0.15s",
                fontFamily: "inherit",
              }}
              onMouseEnter={(event) => {
                event.currentTarget.style.color = d ? "#c8a87a" : "#6a5a4a";
              }}
              onMouseLeave={(event) => {
                event.currentTarget.style.color = d ? "#5a4e3a" : "#b0a090";
              }}
            >
              &times;
            </button>

            <div
              style={{
                fontSize: 10,
                letterSpacing: "0.28em",
                textTransform: "uppercase",
                color: d ? "#5a4e3a" : "#b0a090",
                marginBottom: 10,
              }}
            >
              Replaced
            </div>

            <div
              style={{
                display: "grid",
                gridTemplateColumns: "1fr 1fr",
                gap: 12,
                marginBottom: 14,
              }}
            >
              <button
                onClick={() => {
                  if (
                    !toggledChanges.has(activeChange.key) &&
                    activeChange.resolvedStart !== null &&
                    activeChange.resolvedEnd !== null
                  ) {
                    toggleChange(activeChange);
                  }
                }}
                disabled={activeChange.resolvedStart === null || activeChange.resolvedEnd === null}
                style={{
                  background: toggledChanges.has(activeChange.key) ? theme.hoverBg : "transparent",
                  border: "1px solid " + (toggledChanges.has(activeChange.key) ? accent : theme.modalBorder),
                  color: toggledChanges.has(activeChange.key) ? (d ? "#c8a87a" : "#8a5a20") : d ? "#6a5a48" : "#a09080",
                  padding: "12px 14px",
                  textAlign: "left",
                  cursor:
                    activeChange.resolvedStart === null || activeChange.resolvedEnd === null
                      ? "default"
                      : "pointer",
                  opacity:
                    activeChange.resolvedStart === null || activeChange.resolvedEnd === null
                      ? 0.5
                      : 1,
                  transition: "border-color 0.15s, background 0.15s, color 0.15s",
                  fontFamily: "inherit",
                }}
              >
                <div
                  style={{
                    fontSize: 9,
                    letterSpacing: "0.18em",
                    textTransform: "uppercase",
                    color: toggledChanges.has(activeChange.key) ? theme.accentText : theme.textFaint,
                    marginBottom: 8,
                  }}
                >
                  Original
                </div>
                <div
                  style={{
                    fontFamily: "Palatino, serif",
                    fontStyle: "italic",
                    fontSize: 18,
                    lineHeight: 1.45,
                  }}
                >
                  {activeChange.original}
                </div>
              </button>

              <button
                onClick={() => {
                  if (
                    toggledChanges.has(activeChange.key) &&
                    activeChange.resolvedStart !== null &&
                    activeChange.resolvedEnd !== null
                  ) {
                    toggleChange(activeChange);
                  }
                }}
                disabled={activeChange.resolvedStart === null || activeChange.resolvedEnd === null}
                style={{
                  background: toggledChanges.has(activeChange.key) ? "transparent" : theme.hoverBg,
                  border: "1px solid " + (!toggledChanges.has(activeChange.key) ? accent : theme.modalBorder),
                  color: toggledChanges.has(activeChange.key) ? (d ? "#6a5a48" : "#a09080") : d ? "#c8a87a" : "#8a5a20",
                  padding: "12px 14px",
                  textAlign: "left",
                  cursor:
                    activeChange.resolvedStart === null || activeChange.resolvedEnd === null
                      ? "default"
                      : "pointer",
                  opacity:
                    activeChange.resolvedStart === null || activeChange.resolvedEnd === null
                      ? 0.5
                      : 1,
                  transition: "border-color 0.15s, background 0.15s, color 0.15s",
                  fontFamily: "inherit",
                }}
              >
                <div
                  style={{
                    fontSize: 9,
                    letterSpacing: "0.18em",
                    textTransform: "uppercase",
                    color: toggledChanges.has(activeChange.key) ? theme.textFaint : theme.accentText,
                    marginBottom: 8,
                  }}
                >
                  Edited
                </div>
                <div
                  style={{
                    fontFamily: "Palatino, serif",
                    fontStyle: "italic",
                    fontSize: 18,
                    lineHeight: 1.45,
                  }}
                >
                  {activeChange.replacement}
                </div>
              </button>
            </div>

            <div
              style={{
                fontSize: 10,
                textTransform: "uppercase",
                letterSpacing: "0.18em",
                color: theme.textFaint,
                marginBottom: 12,
              }}
            >
              {toggledChanges.has(activeChange.key) ? "Showing original wording" : "Showing edited wording"}
            </div>

            <div
              style={{
                width: "100%",
                height: 1,
                background: theme.modalBorder,
                marginBottom: 12,
              }}
            />

            <div
              style={{
                fontSize: 13,
                lineHeight: 1.6,
                color: d ? "#9a8a78" : "#6a5a48",
              }}
            >
              {activeChange.reason}
            </div>
          </div>
        </div>
      ) : null}

      <style dangerouslySetInnerHTML={{ __html: cssText }} />
    </div>
  );
}

export default App;
