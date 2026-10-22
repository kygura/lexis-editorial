import { memo, useEffect, useLayoutEffect, useMemo, useReducer, useRef, useState } from "react";
import "./Folio.css";
import {
  KEY_ERROR_CODES,
  applyChangeToggles,
  buildSegments,
  extractPartialRewrite,
  parseEditorJson,
  readEditorStream,
  sanitizeChanges,
  toLexisError,
  type AnalysisResult,
  type LexisError,
} from "./lib/proof";
import { blockKind, splitMarkdown, wordCount } from "./lib/markdown";

// Pause after the last keystroke before a paragraph goes to the editor.
const DEBOUNCE_MS = 1100;
const MAX_IN_FLIGHT = 3;
const MIN_LENGTH = 8;
const CONTEXT_BEFORE = 1500;
const CONTEXT_AFTER = 600;
const RATE_LIMIT_PAUSE_MS = 8000;
const RATE_LIMIT_RETRIES = 3;
const CONFIRM_OVER = 25;
const SAVE_THROTTLE_MS = 1000;
const STORAGE_KEY = "lexis-folio";
const LIVE_KEY = "lexis-folio-live";
const MARKDOWN_FILE = /\.(md|markdown|mdown|mkd|txt)$/i;

export interface FolioSettings {
  apiUrl: string;
  provider: string;
  model: string;
  effort: string;
  apiKey: string;
  styleDirective: string;
}

interface FolioProps {
  settings: FolioSettings;
  onStyleDirectiveChange: (value: string) => void;
  onOpenKeys: () => void;
}

type BlockStatus = "idle" | "waiting" | "queued" | "streaming" | "error";

interface Block {
  id: string;
  text: string;
  // The file the block was imported from, if any.
  source?: string;
  proof: AnalysisResult | null;
  // The manuscript text `proof` was made from; differs from `text` once edited.
  proofText: string;
  // Change keys restored to the original wording.
  stet: string[];
  // Export the manuscript rather than the proof.
  keep: boolean;
  status: BlockStatus;
  preview: string;
  error?: LexisError;
}

type Action =
  | { type: "edit"; id: string; text: string; pending: boolean }
  | { type: "replace"; id: string; blocks: Array<{ id: string; text: string }>; pending: boolean }
  | { type: "merge"; id: string }
  | { type: "import"; blocks: Block[] }
  | { type: "status"; id: string; status: BlockStatus; error?: LexisError }
  | { type: "preview"; id: string; preview: string }
  | { type: "proofed"; id: string; text: string; proof: AnalysisResult }
  | { type: "stet"; id: string; key: string }
  | { type: "keep"; id: string }
  | { type: "accept"; id: string; text: string }
  | { type: "settle" }
  | { type: "clear" };

function newId() {
  return typeof crypto !== "undefined" && "randomUUID" in crypto
    ? crypto.randomUUID()
    : Math.random().toString(36).slice(2) + Date.now().toString(36);
}

function makeBlock(text: string, source?: string, id = newId()): Block {
  return { id, text, source, proof: null, proofText: "", stet: [], keep: false, status: "idle", preview: "" };
}

function isProofable(text: string) {
  return blockKind(text) === "prose" && text.trim().length >= MIN_LENGTH;
}

function isFresh(block: Block) {
  return !!block.proof && block.proofText === block.text;
}

function effectiveText(block: Block) {
  if (block.keep || !block.proof || !isFresh(block) || blockKind(block.text) !== "prose") return block.text;
  const changes = sanitizeChanges(block.proof.rewrite, block.proof.changes ?? []);
  return applyChangeToggles(block.proof.rewrite, changes, new Set(block.stet));
}

function update(state: Block[], id: string, patch: (block: Block) => Block) {
  const index = state.findIndex((block) => block.id === id);
  if (index === -1) return state;
  const next = state.slice();
  next[index] = patch(state[index]);
  return next;
}

function reducer(state: Block[], action: Action): Block[] {
  switch (action.type) {
    case "edit":
      return update(state, action.id, (block) => ({
        ...block,
        text: action.text,
        status: action.pending ? "waiting" : "idle",
        preview: "",
        error: undefined,
      }));
    case "replace": {
      const index = state.findIndex((block) => block.id === action.id);
      if (index === -1) return state;
      const original = state[index];
      const replacements = action.blocks.map(({ id, text }) => {
        const base = id === original.id ? original : makeBlock("", original.source, id);
        return {
          ...base,
          text,
          status: (action.pending && isProofable(text) ? "waiting" : "idle") as BlockStatus,
          preview: "",
          error: undefined,
        };
      });
      return [...state.slice(0, index), ...replacements, ...state.slice(index + 1)];
    }
    case "merge": {
      const index = state.findIndex((block) => block.id === action.id);
      if (index <= 0) return state;
      const previous = state[index - 1];
      const merged = { ...previous, text: previous.text + state[index].text, status: "idle" as BlockStatus };
      return [...state.slice(0, index - 1), merged, ...state.slice(index + 1)];
    }
    case "import": {
      const kept = state.some((block) => block.text.trim()) ? state : [];
      return [...kept, ...action.blocks];
    }
    case "status":
      return update(state, action.id, (block) => ({
        ...block,
        status: action.status,
        preview: action.status === "streaming" ? "" : block.preview,
        error: action.error,
      }));
    case "preview":
      return update(state, action.id, (block) => ({ ...block, preview: action.preview }));
    case "proofed":
      return update(state, action.id, (block) =>
        block.text === action.text
          ? { ...block, proof: action.proof, proofText: action.text, stet: [], status: "idle", preview: "", error: undefined }
          : block,
      );
    case "stet":
      return update(state, action.id, (block) => ({
        ...block,
        stet: block.stet.includes(action.key)
          ? block.stet.filter((key) => key !== action.key)
          : [...block.stet, action.key],
      }));
    case "keep":
      return update(state, action.id, (block) => ({ ...block, keep: !block.keep }));
    case "accept":
      // The proof becomes the manuscript; recording it as already proofed
      // keeps the scheduler from sending the editor its own work.
      return update(state, action.id, (block) => ({
        ...block,
        text: action.text,
        proof: block.proof ? { ...block.proof, rewrite: action.text, changes: [] } : null,
        proofText: action.text,
        stet: [],
        keep: false,
        status: "idle",
        preview: "",
        error: undefined,
      }));
    case "settle":
      return state.map((block) =>
        block.status === "waiting" || block.status === "queued" ? { ...block, status: "idle" } : block,
      );
    case "clear":
      return [makeBlock("")];
  }
}

function loadBlocks(): Block[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    const saved = raw ? (JSON.parse(raw) as Array<Partial<Block>>) : [];
    const blocks = saved
      .filter((block) => typeof block.text === "string")
      .map((block) => ({
        ...makeBlock(block.text!, block.source, block.id || newId()),
        proof: block.proof ?? null,
        proofText: block.proofText ?? "",
        stet: Array.isArray(block.stet) ? block.stet : [],
        keep: !!block.keep,
      }));
    return blocks.length ? blocks : [makeBlock("")];
  } catch {
    return [makeBlock("")];
  }
}

function saveBlocks(blocks: Block[]) {
  const persisted = blocks.map(({ id, text, source, proof, proofText, stet, keep }) => ({
    id,
    text,
    source,
    proof,
    proofText,
    stet,
    keep,
  }));
  localStorage.setItem(STORAGE_KEY, JSON.stringify(persisted));
}

function settingsKey(settings: FolioSettings) {
  return JSON.stringify([settings.provider, settings.model, settings.effort, settings.styleDirective.trim()]);
}

function contextBefore(blocks: Block[], index: number) {
  const parts: string[] = [];
  let length = 0;
  for (let i = index - 1; i >= 0 && length < CONTEXT_BEFORE; i--) {
    const text = blocks[i].text.trim();
    if (!text) continue;
    parts.unshift(text);
    length += text.length + 2;
  }
  return parts.join("\n\n").slice(-CONTEXT_BEFORE);
}

function contextAfter(blocks: Block[], index: number) {
  for (let i = index + 1; i < blocks.length; i++) {
    const text = blocks[i].text.trim();
    if (text) return text.slice(0, CONTEXT_AFTER);
  }
  return "";
}

function readStorage(key: string) {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

interface FolioActions {
  register: (id: string, element: HTMLTextAreaElement | null) => void;
  edit: (id: string, text: string) => void;
  keyDown: (id: string, event: React.KeyboardEvent<HTMLTextAreaElement>) => void;
  paste: (id: string, event: React.ClipboardEvent<HTMLTextAreaElement>) => void;
  blur: (id: string) => void;
  proofNow: (id: string) => void;
  stet: (id: string, key: string) => void;
  keep: (id: string) => void;
  accept: (id: string) => void;
  openKeys: () => void;
}

export default function Folio({ settings, onStyleDirectiveChange, onOpenKeys }: FolioProps) {
  const [blocks, dispatch] = useReducer(reducer, undefined, loadBlocks);
  const [live, setLive] = useState(() => readStorage(LIVE_KEY) !== "false");
  const [notice, setNotice] = useState<LexisError | null>(null);
  const [imported, setImported] = useState<string | null>(null);
  const [saveFailed, setSaveFailed] = useState(false);
  const [dragging, setDragging] = useState(false);

  const blocksRef = useRef(blocks);
  const settingsRef = useRef(settings);
  const liveRef = useRef(live);
  const openKeysRef = useRef(onOpenKeys);
  blocksRef.current = blocks;
  settingsRef.current = settings;
  liveRef.current = live;
  openKeysRef.current = onOpenKeys;

  const timers = useRef(new Map<string, number>());
  const queue = useRef<string[]>([]);
  const forced = useRef(new Set<string>());
  const inFlight = useRef(new Map<string, AbortController>());
  const cache = useRef(new Map<string, AnalysisResult>());
  const retries = useRef(new Map<string, number>());
  const pausedUntil = useRef(0);
  const inputs = useRef(new Map<string, HTMLTextAreaElement>());
  const pendingFocus = useRef<{ id: string; caret: number } | null>(null);
  const saveTimer = useRef<number | null>(null);
  const fileInput = useRef<HTMLInputElement | null>(null);
  const folderInput = useRef<HTMLInputElement | null>(null);

  // The scheduler reads everything through refs, so the actions handed to
  // each row stay stable and memoised rows only re-render when their block does.
  const engine = useMemo(() => {
    const clearTimer = (id: string) => {
      const timer = timers.current.get(id);
      if (timer !== undefined) {
        window.clearTimeout(timer);
        timers.current.delete(id);
      }
    };

    const cancel = (id: string) => {
      clearTimer(id);
      queue.current = queue.current.filter((queued) => queued !== id);
      forced.current.delete(id);
      const controller = inFlight.current.get(id);
      if (controller) {
        controller.abort();
        inFlight.current.delete(id);
      }
    };

    const halt = (error: LexisError) => {
      for (const id of timers.current.keys()) window.clearTimeout(timers.current.get(id));
      timers.current.clear();
      queue.current = [];
      forced.current.clear();
      dispatch({ type: "settle" });
      setNotice(error);
    };

    const pump = () => {
      if (Date.now() < pausedUntil.current) return;
      while (inFlight.current.size < MAX_IN_FLIGHT && queue.current.length > 0) {
        const id = queue.current.shift()!;
        const force = forced.current.delete(id);
        void run(id, force);
      }
    };

    const enqueue = (id: string, force = false) => {
      clearTimer(id);
      if (force) {
        forced.current.add(id);
        const controller = inFlight.current.get(id);
        if (controller) {
          controller.abort();
          inFlight.current.delete(id);
        }
      }
      if (!queue.current.includes(id) && !inFlight.current.has(id)) {
        queue.current.push(id);
        dispatch({ type: "status", id, status: "queued" });
      }
      pump();
    };

    const schedule = (id: string) => {
      clearTimer(id);
      timers.current.set(
        id,
        window.setTimeout(() => {
          timers.current.delete(id);
          enqueue(id);
        }, DEBOUNCE_MS),
      );
    };

    const run = async (id: string, force: boolean) => {
      const list = blocksRef.current;
      const index = list.findIndex((block) => block.id === id);
      const block = list[index];
      if (!block || !isProofable(block.text) || (!force && isFresh(block))) {
        if (block) dispatch({ type: "status", id, status: "idle" });
        pump();
        return;
      }

      const current = settingsRef.current;
      const text = block.text;
      const cacheKey = settingsKey(current) + "\u0000" + text.trim();
      const cached = force ? undefined : cache.current.get(cacheKey);
      if (cached) {
        dispatch({ type: "proofed", id, text, proof: cached });
        pump();
        return;
      }

      const controller = new AbortController();
      inFlight.current.set(id, controller);
      dispatch({ type: "status", id, status: "streaming" });

      try {
        const response = await fetch(current.apiUrl + "/analyze", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          signal: controller.signal,
          body: JSON.stringify({
            text: text.trim(),
            style_directive: current.styleDirective,
            provider: current.provider,
            model: current.model,
            effort: current.effort,
            api_key: current.apiKey || undefined,
            context_before: contextBefore(list, index),
            context_after: contextAfter(list, index),
          }),
        });

        let lastPreview = "";
        const fullText = await readEditorStream(response, (soFar) => {
          const preview = extractPartialRewrite(soFar);
          if (preview !== lastPreview) {
            lastPreview = preview;
            dispatch({ type: "preview", id, preview });
          }
        });

        const proof = parseEditorJson<AnalysisResult>(fullText);
        cache.current.set(cacheKey, proof);
        retries.current.delete(id);
        dispatch({ type: "proofed", id, text, proof });
      } catch (caught) {
        if (controller.signal.aborted) return;
        const error = toLexisError(caught, "This paragraph didn't come back from the editor. Try again.");
        const attempts = retries.current.get(id) ?? 0;
        if (error.code === "rate_limit" && attempts < RATE_LIMIT_RETRIES) {
          retries.current.set(id, attempts + 1);
          queue.current.unshift(id);
          if (force) forced.current.add(id);
          dispatch({ type: "status", id, status: "queued" });
          pausedUntil.current = Date.now() + RATE_LIMIT_PAUSE_MS;
          window.setTimeout(pump, RATE_LIMIT_PAUSE_MS);
          return;
        }
        dispatch({ type: "status", id, status: "error", error });
        // A bad key fails every paragraph the same way; stop the press.
        if (error.code && KEY_ERROR_CODES.has(error.code)) halt(error);
      } finally {
        if (inFlight.current.get(id) === controller) inFlight.current.delete(id);
        pump();
      }
    };

    const focus = (id: string, caret: number) => {
      const element = inputs.current.get(id);
      if (!element) return;
      element.focus();
      element.setSelectionRange(caret, caret);
    };

    // Swaps one block for several (a split or a multi-paragraph paste).
    const replace = (id: string, next: Array<{ id: string; text: string }>, focusTarget: { id: string; caret: number }) => {
      const pending = liveRef.current;
      cancel(id);
      pendingFocus.current = focusTarget;
      dispatch({ type: "replace", id, blocks: next, pending });
      if (pending) next.forEach((block) => isProofable(block.text) && schedule(block.id));
    };

    const actions: FolioActions = {
      register(id, element) {
        if (element) inputs.current.set(id, element);
        else inputs.current.delete(id);
      },
      edit(id, text) {
        const pending = liveRef.current && isProofable(text);
        cancel(id);
        dispatch({ type: "edit", id, text, pending });
        if (pending) schedule(id);
      },
      keyDown(id, event) {
        if (event.nativeEvent.isComposing) return;
        const list = blocksRef.current;
        const index = list.findIndex((block) => block.id === id);
        if (index === -1) return;
        const element = event.currentTarget;
        const { selectionStart: start, selectionEnd: end, value } = element;
        const plain = !event.shiftKey && !event.altKey && !event.ctrlKey && !event.metaKey;

        if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
          event.preventDefault();
          enqueue(id, true);
          return;
        }

        if (event.key === "Enter" && plain && blockKind(value) !== "code") {
          event.preventDefault();
          const nextId = newId();
          replace(
            id,
            [
              { id, text: value.slice(0, start).replace(/\s+$/, "") },
              { id: nextId, text: value.slice(end).replace(/^\s+/, "") },
            ],
            { id: nextId, caret: 0 },
          );
          return;
        }

        if (event.key === "Backspace" && plain && start === 0 && end === 0 && index > 0) {
          event.preventDefault();
          const previous = list[index - 1];
          cancel(id);
          cancel(previous.id);
          pendingFocus.current = { id: previous.id, caret: previous.text.length };
          dispatch({ type: "merge", id });
          if (liveRef.current && isProofable(previous.text + value)) schedule(previous.id);
          return;
        }

        if (event.key === "ArrowUp" && plain && start === 0 && end === 0 && index > 0) {
          event.preventDefault();
          const previous = list[index - 1];
          focus(previous.id, previous.text.length);
          return;
        }

        if (event.key === "ArrowDown" && plain && start === value.length && end === start && index < list.length - 1) {
          event.preventDefault();
          focus(list[index + 1].id, 0);
        }
      },
      paste(id, event) {
        const pasted = event.clipboardData.getData("text/plain");
        if (!/\n\s*\n/.test(pasted)) return;
        event.preventDefault();
        const element = event.currentTarget;
        const before = element.value.slice(0, element.selectionStart);
        const after = element.value.slice(element.selectionEnd);
        const pieces = splitMarkdown(before + pasted);
        if (!pieces.length) return;
        const caret = pieces[pieces.length - 1].length;
        pieces[pieces.length - 1] += after;
        const next = pieces.map((text, i) => ({ id: i === 0 ? id : newId(), text }));
        replace(id, next, { id: next[next.length - 1].id, caret });
      },
      blur(id) {
        // Leaving a paragraph sends it without waiting out the pause.
        if (timers.current.has(id)) enqueue(id);
      },
      proofNow(id) {
        enqueue(id, true);
      },
      stet(id, key) {
        dispatch({ type: "stet", id, key });
      },
      keep(id) {
        dispatch({ type: "keep", id });
      },
      accept(id) {
        const block = blocksRef.current.find((candidate) => candidate.id === id);
        if (!block?.proof) return;
        cancel(id);
        dispatch({ type: "accept", id, text: effectiveText({ ...block, keep: false }) });
      },
      openKeys() {
        openKeysRef.current();
      },
    };

    const stopAll = () => {
      for (const timer of timers.current.values()) window.clearTimeout(timer);
      timers.current.clear();
      queue.current = [];
      forced.current.clear();
      for (const controller of inFlight.current.values()) controller.abort();
      inFlight.current.clear();
    };

    const resetScheduler = () => {
      stopAll();
      pausedUntil.current = 0;
      retries.current.clear();
      dispatch({ type: "settle" });
    };

    return { actions, enqueue, stopAll, resetScheduler, cancel };
  }, []);

  useLayoutEffect(() => {
    const target = pendingFocus.current;
    if (!target) return;
    const element = inputs.current.get(target.id);
    if (!element) return;
    pendingFocus.current = null;
    element.focus();
    element.setSelectionRange(target.caret, target.caret);
  }, [blocks]);

  // Throttled save: a long run of streamed paragraphs still gets written.
  useEffect(() => {
    if (saveTimer.current !== null) return;
    saveTimer.current = window.setTimeout(() => {
      saveTimer.current = null;
      try {
        saveBlocks(blocksRef.current);
        setSaveFailed(false);
      } catch {
        setSaveFailed(true);
      }
    }, SAVE_THROTTLE_MS);
  }, [blocks]);

  useEffect(() => {
    const flushSave = () => {
      try {
        saveBlocks(blocksRef.current);
      } catch {
        // Already surfaced by the throttled save.
      }
    };
    window.addEventListener("pagehide", flushSave);
    return () => {
      window.removeEventListener("pagehide", flushSave);
      if (saveTimer.current !== null) {
        window.clearTimeout(saveTimer.current);
        saveTimer.current = null;
      }
      flushSave();
      engine.stopAll();
    };
  }, [engine]);

  useEffect(() => {
    try {
      localStorage.setItem(LIVE_KEY, String(live));
    } catch {
      // Live stays on for this visit.
    }
  }, [live]);

  useEffect(() => {
    folderInput.current?.setAttribute("webkitdirectory", "");
  }, []);

  useEffect(() => {
    if (!imported) return;
    const timer = window.setTimeout(() => setImported(null), 6000);
    return () => window.clearTimeout(timer);
  }, [imported]);

  const stats = useMemo(() => {
    let paragraphs = 0;
    let words = 0;
    let set = 0;
    let stale = 0;
    let pending = 0;
    let pressing = 0;
    let failed = 0;
    let unproofed = 0;
    for (const block of blocks) {
      words += wordCount(block.text);
      if (!isProofable(block.text)) continue;
      paragraphs++;
      if (block.status === "streaming") pressing++;
      else if (block.status === "waiting" || block.status === "queued") pending++;
      else if (block.status === "error") failed++;
      if (isFresh(block)) set++;
      else {
        unproofed++;
        if (block.proof) stale++;
      }
    }
    return { paragraphs, words, set, stale, pending, pressing, failed, unproofed };
  }, [blocks]);

  const toggleLive = () => {
    const next = !live;
    setLive(next);
    liveRef.current = next;
    if (!next) engine.resetScheduler();
  };

  const proofAll = () => {
    const ids = blocksRef.current.filter((block) => isProofable(block.text) && !isFresh(block)).map((block) => block.id);
    if (!ids.length) return;
    if (ids.length > CONFIRM_OVER && !window.confirm(`Send ${ids.length} paragraphs to the editor? Each one is a separate request.`)) {
      return;
    }
    setNotice(null);
    ids.forEach((id) => engine.enqueue(id));
  };

  const importFiles = async (files: FileList | File[] | null) => {
    const list = Array.from(files ?? [])
      .filter((file) => MARKDOWN_FILE.test(file.name))
      .sort((a, b) =>
        (a.webkitRelativePath || a.name).localeCompare(b.webkitRelativePath || b.name, undefined, { numeric: true }),
      );
    if (!list.length) {
      setImported("Nothing to import. Lexis reads .md and .txt files.");
      return;
    }
    const texts = await Promise.all(list.map((file) => file.text()));
    const incoming = list.flatMap((file, i) =>
      splitMarkdown(texts[i]).map((text) => makeBlock(text, file.webkitRelativePath || file.name)),
    );
    dispatch({ type: "import", blocks: incoming });
    const words = incoming.reduce((sum, block) => sum + wordCount(block.text), 0);
    setImported(
      `Imported ${list.length} file${list.length !== 1 ? "s" : ""} · ${incoming.length.toLocaleString()} blocks · ${words.toLocaleString()} words`,
    );
  };

  const exportMarkdown = () => {
    const markdown = blocksRef.current
      .filter((block) => block.text.trim())
      .map(effectiveText)
      .join("\n\n");
    const blob = new Blob([markdown + "\n"], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `lexis-folio-${new Date().toISOString().slice(0, 10)}.md`;
    link.click();
    URL.revokeObjectURL(url);
  };

  const clearFolio = () => {
    if (!window.confirm("Clear the folio? Export first if you want to keep it.")) return;
    engine.resetScheduler();
    cache.current.clear();
    setNotice(null);
    dispatch({ type: "clear" });
  };

  const busy = stats.pressing > 0 || stats.pending > 0;
  const statusLine = [
    stats.pressing ? `${stats.pressing} in press` : "",
    stats.pending ? `${stats.pending} queued` : "",
    stats.set ? `${stats.set} set` : "",
    stats.stale ? `${stats.stale} edited` : "",
    stats.failed ? `${stats.failed} failed` : "",
  ]
    .filter(Boolean)
    .join(" · ");

  return (
    <section
      className="lx-sheet lx-folio"
      aria-label="Folio"
      onDragOver={(event) => {
        if (!event.dataTransfer.types.includes("Files")) return;
        event.preventDefault();
        setDragging(true);
      }}
      onDragLeave={(event) => {
        if (event.currentTarget === event.target) setDragging(false);
      }}
      onDrop={(event) => {
        if (!event.dataTransfer.files.length) return;
        event.preventDefault();
        setDragging(false);
        void importFiles(event.dataTransfer.files);
      }}
    >
      <div className="lx-folio-bar">
        <div className="lx-folio-bar-row">
          <span className="lx-eyebrow">Folio</span>
          <span className="lx-meta">
            {stats.paragraphs.toLocaleString()} paragraph{stats.paragraphs !== 1 ? "s" : ""} ·{" "}
            {stats.words.toLocaleString()} words
          </span>
          <span className="lx-toolbar-spacer" />
          <span className={"lx-status" + (busy ? " is-busy" : "")} aria-live="polite">
            {statusLine}
          </span>
        </div>

        <div className="lx-folio-bar-row lx-folio-controls">
          <button
            role="switch"
            aria-checked={live}
            title={live ? "Paragraphs go to the editor as you pause" : "Paragraphs wait until you proof them"}
            className={"lx-effort-key" + (live ? " is-on" : "")}
            onClick={toggleLive}
          >
            {live ? "Live" : "Paused"}
          </button>
          <button className="lx-link" onClick={() => fileInput.current?.click()}>
            Import .md
          </button>
          <button className="lx-link" onClick={() => folderInput.current?.click()}>
            Import folder
          </button>
          <button className="lx-link" onClick={proofAll} disabled={stats.unproofed === 0}>
            Proof all{stats.unproofed ? ` (${stats.unproofed.toLocaleString()})` : ""}
          </button>
          <button className="lx-link" onClick={exportMarkdown} disabled={stats.words === 0}>
            Export .md
          </button>
          <button className="lx-link" onClick={clearFolio} disabled={stats.words === 0}>
            Clear
          </button>
          <label className="lx-folio-brief">
            <span className="lx-chip-label">Brief</span>
            <input
              className="lx-input"
              value={settings.styleDirective}
              onChange={(event) => onStyleDirectiveChange(event.target.value)}
              placeholder="Steer voice, cadence, constraints…"
            />
          </label>
          <input
            ref={fileInput}
            type="file"
            accept=".md,.markdown,.mdown,.mkd,.txt,text/markdown,text/plain"
            multiple
            hidden
            onChange={(event) => {
              void importFiles(event.target.files);
              event.target.value = "";
            }}
          />
          <input
            ref={folderInput}
            type="file"
            multiple
            hidden
            onChange={(event) => {
              void importFiles(event.target.files);
              event.target.value = "";
            }}
          />
        </div>

        {imported ? <p className="lx-meta lx-folio-note">{imported}</p> : null}
        {saveFailed ? (
          <p className="lx-meta lx-folio-note is-warning">
            This folio is too large to keep in the browser. Export it to keep your work.
          </p>
        ) : null}
        {notice ? (
          <div className="lx-error lx-folio-notice" role="alert">
            <p>{notice.message} The press has stopped.</p>
            <span className="lx-folio-notice-actions">
              {notice.code && KEY_ERROR_CODES.has(notice.code) ? (
                <button className="lx-link" onClick={onOpenKeys}>
                  Open keys
                </button>
              ) : null}
              <button className="lx-link" onClick={() => setNotice(null)}>
                Dismiss
              </button>
            </span>
          </div>
        ) : null}
      </div>

      <div className="lx-folio-columns" aria-hidden>
        <span className="lx-eyebrow">Manuscript</span>
        <span />
        <span className="lx-eyebrow">Proof</span>
      </div>

      <div className="lx-folio-body">
        {blocks.map((block, index) => (
          <FolioRow
            key={block.id}
            block={block}
            source={block.source && block.source !== blocks[index - 1]?.source ? block.source : undefined}
            placeholder={blocks.length === 1 ? "Begin typing, or drop Markdown files here…" : undefined}
            actions={engine.actions}
          />
        ))}
      </div>

      {dragging ? (
        <div className="lx-folio-drop" aria-hidden>
          <span>Drop Markdown files to add them to the folio</span>
        </div>
      ) : null}
    </section>
  );
}

interface FolioRowProps {
  block: Block;
  source?: string;
  placeholder?: string;
  actions: FolioActions;
}

const FolioRow = memo(function FolioRow({ block, source, placeholder, actions }: FolioRowProps) {
  const kind = blockKind(block.text);
  const fresh = isFresh(block);
  const streaming = block.status === "streaming";
  const changes = useMemo(
    () => (block.proof ? sanitizeChanges(block.proof.rewrite, block.proof.changes ?? []) : []),
    [block.proof],
  );
  const stet = useMemo(() => new Set(block.stet), [block.stet]);
  const wide = kind === "heading" || kind === "code" || kind === "rule";

  let state = "";
  if (streaming) state = "Setting";
  else if (block.status === "queued") state = "Queued";
  else if (block.status === "waiting") state = "·";
  else if (block.status === "error") state = "Failed";
  else if (fresh) state = changes.length ? `${changes.length} mark${changes.length !== 1 ? "s" : ""}` : "Clean";
  else if (block.proof) state = "Edited";

  const proofText = () => {
    if (streaming && block.preview) {
      return (
        <p className="lx-folio-proof lx-setting">
          {block.preview}
          <span className="lx-cursor" aria-hidden />
        </p>
      );
    }
    if (block.proof) {
      const segments = buildSegments(block.proof.rewrite, changes);
      return (
        <p className={"lx-folio-proof" + (fresh ? "" : " is-stale") + (block.keep ? " is-kept" : "")}>
          {segments.map((segment, index) => {
            if (segment.type === "plain") return <span key={index}>{segment.text}</span>;
            const change = segment.change!;
            const restored = stet.has(change.key);
            return (
              <span
                key={change.key}
                role="button"
                tabIndex={0}
                className={"lx-mark" + (restored ? " is-restored" : "")}
                title={`${change.original} → ${change.replacement}${change.reason ? `\n${change.reason}` : ""}\n${restored ? "Click to use the edit" : "Click to restore the original"}`}
                onClick={() => actions.stet(block.id, change.key)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault();
                    actions.stet(block.id, change.key);
                  }
                }}
              >
                {restored ? change.original : segment.text}
              </span>
            );
          })}
          {streaming ? <span className="lx-cursor" aria-hidden /> : null}
        </p>
      );
    }
    if (streaming) {
      return (
        <p className="lx-folio-proof lx-setting">
          <span className="lx-cursor" aria-hidden />
        </p>
      );
    }
    return null;
  };

  return (
    <div className={`lx-folio-row is-${kind}` + (wide ? " is-wide" : "") + (streaming ? " is-streaming" : "")}>
      {source ? (
        <div className="lx-folio-source">
          <span>{source}</span>
        </div>
      ) : null}

      <div className="lx-folio-grow" data-value={block.text}>
        <textarea
          ref={(element) => actions.register(block.id, element)}
          rows={1}
          spellCheck
          className="lx-folio-input"
          value={block.text}
          placeholder={placeholder}
          aria-label="Paragraph"
          onChange={(event) => actions.edit(block.id, event.target.value)}
          onKeyDown={(event) => actions.keyDown(block.id, event)}
          onPaste={(event) => actions.paste(block.id, event)}
          onBlur={() => actions.blur(block.id)}
        />
      </div>

      {wide ? null : (
        <>
          <button
            className={"lx-folio-state" + (streaming ? " is-busy" : "") + (block.status === "error" ? " is-error" : "")}
            disabled={!isProofable(block.text)}
            title={isProofable(block.text) ? "Proof this paragraph again (Ctrl/Cmd + Enter)" : undefined}
            onClick={() => actions.proofNow(block.id)}
          >
            {state}
          </button>

          <div className="lx-folio-out">
            {proofText()}

            {block.status === "error" && block.error ? (
              <p className="lx-card-error">
                {block.error.message}{" "}
                {block.error.code && KEY_ERROR_CODES.has(block.error.code) ? (
                  <button className="lx-link lx-link-small" onClick={actions.openKeys}>
                    Open keys
                  </button>
                ) : (
                  <button className="lx-link lx-link-small" onClick={() => actions.proofNow(block.id)}>
                    Retry
                  </button>
                )}
              </p>
            ) : null}

            {fresh && !streaming && changes.length > 0 ? (
              <div className="lx-folio-row-actions">
                <button className="lx-link lx-link-small" onClick={() => actions.accept(block.id)}>
                  Accept into manuscript
                </button>
                <button className="lx-link lx-link-small" onClick={() => actions.keep(block.id)}>
                  {block.keep ? "Use the proof" : "Keep original"}
                </button>
              </div>
            ) : null}
          </div>
        </>
      )}
    </div>
  );
});
