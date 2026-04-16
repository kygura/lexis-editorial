import { useState, useRef, useEffect, useCallback } from "react";
import "./App.css";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";
const DEBOUNCE_MS = 900;
const MIN_LENGTH = 8;
const SIMILARITY_THRESHOLD = 0.85;

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

interface Change {
  original: string;
  replacement: string;
  start: number;
  end: number;
  reason: string;
}

interface Analysis {
  register: string;
  tone: string;
  notes?: string;
}

interface AnalysisResult {
  analysis: Analysis;
  rewrite: string;
  changes: Change[];
}

interface Segment {
  type: "plain" | "highlight";
  text: string;
  change?: Change;
}

function useDebounce(value: string, delay: number) {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const t = setTimeout(() => setDebounced(value), delay);
    return () => clearTimeout(t);
  }, [value, delay]);
  return debounced;
}

function buildSegments(rewrite: string, changes: Change[]): Segment[] {
  if (!changes?.length) return [{ type: "plain", text: rewrite }];
  const sorted = [...changes].sort((a, b) => a.start - b.start);
  const segments: Segment[] = [];
  let cursor = 0;
  for (const change of sorted) {
    if (change.start < cursor) continue;
    if (change.start > cursor) {
      segments.push({ type: "plain", text: rewrite.slice(cursor, change.start) });
    }
    segments.push({
      type: "highlight",
      text: rewrite.slice(change.start, change.end),
      change,
    });
    cursor = change.end;
  }
  if (cursor < rewrite.length) {
    segments.push({ type: "plain", text: rewrite.slice(cursor) });
  }
  return segments;
}

function App() {
  const [input, setInput] = useState("");
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [streaming, setStreaming] = useState(false);
  const [activeChange, setActiveChange] = useState<Change | null>(null);
  const [modalPos, setModalPos] = useState<{ x: number; y: number } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dark, setDark] = useState(true);
  const [copied, setCopied] = useState(false);
  const [toggledChanges, setToggledChanges] = useState<Set<number>>(new Set());
  const [styleDirective, setStyleDirective] = useState("");
  const [styleOpen, setStyleOpen] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const debouncedInput = useDebounce(input, DEBOUNCE_MS);
  const debouncedStyle = useDebounce(styleDirective, DEBOUNCE_MS);
  const prevAnalyzed = useRef("");
  const cachedResult = useRef<AnalysisResult | null>(null);
  const cachedInput = useRef("");
  const cachedStyle = useRef("");
  const modalRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const saved = localStorage.getItem("lexis-theme");
    if (saved) setDark(saved === "dark");
    const savedStyle = localStorage.getItem("lexis-style-directive");
    if (savedStyle) setStyleDirective(savedStyle);
  }, []);

  useEffect(() => {
    localStorage.setItem("lexis-theme", dark ? "dark" : "light");
  }, [dark]);

  useEffect(() => {
    localStorage.setItem("lexis-style-directive", styleDirective);
  }, [styleDirective]);

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (modalRef.current && !modalRef.current.contains(e.target as Node)) {
        setActiveChange(null);
        setModalPos(null);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  const analyze = useCallback(async (text: string, force = false) => {
    if (!force && text === prevAnalyzed.current && debouncedStyle === cachedStyle.current) return;
    if (!force && cachedResult.current && cachedInput.current && debouncedStyle === cachedStyle.current) {
      const sim = textSimilarity(text, cachedInput.current);
      if (sim >= SIMILARITY_THRESHOLD) {
        if (abortRef.current) abortRef.current.abort();
        prevAnalyzed.current = text;
        setStreaming(false);
        setResult(cachedResult.current);
        return;
      }
    }
    prevAnalyzed.current = text;

    if (abortRef.current) abortRef.current.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setStreaming(true);
    setResult(null);
    setError(null);
    setActiveChange(null);
    setModalPos(null);
    setToggledChanges(new Set());

    try {
      const response = await fetch(API_URL + "/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: controller.signal,
        body: JSON.stringify({ text, style_directive: debouncedStyle }),
      });

      const reader = response.body!.getReader();
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
              setError(parsed.error);
              return;
            }
            const delta = parsed.text || "";
            if (delta) fullText += delta;
          } catch {
            // skip
          }
        }
      }

      const codeBlockRegex = /```json|```/g;
      const clean = fullText.replace(codeBlockRegex, "").trim();
      const parsed = JSON.parse(clean);
      cachedResult.current = parsed;
      cachedInput.current = text;
      cachedStyle.current = debouncedStyle;
      setResult(parsed);
    } catch (e: unknown) {
      if (e instanceof Error && e.name !== "AbortError") {
        setError("Analysis failed. Try again.");
        prevAnalyzed.current = "";
      }
    } finally {
      setStreaming(false);
    }
  }, [debouncedStyle]);

  useEffect(() => {
    if (debouncedInput.trim().length >= MIN_LENGTH) {
      analyze(debouncedInput.trim());
    } else {
      if (abortRef.current) abortRef.current.abort();
      setResult(null);
      setStreaming(false);
      setActiveChange(null);
      setModalPos(null);
      prevAnalyzed.current = "";
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [debouncedInput, analyze]);

  const copyRewrite = () => {
    if (!result?.rewrite) return;
    let text = result.rewrite;
    if (toggledChanges.size > 0 && result.changes) {
      const sorted = [...result.changes]
        .filter((c) => toggledChanges.has(c.start))
        .sort((a, b) => b.start - a.start);
      for (const change of sorted) {
        text = text.slice(0, change.start) + change.original + text.slice(change.end);
      }
    }
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const toggleChange = (change: Change) => {
    setToggledChanges((prev) => {
      const next = new Set(prev);
      if (next.has(change.start)) {
        next.delete(change.start);
      } else {
        next.add(change.start);
      }
      return next;
    });
  };

  const handleHighlightClick = (change: Change, e: React.MouseEvent) => {
    const rect = (e.target as HTMLElement).getBoundingClientRect();
    if (activeChange?.start === change.start) {
      setActiveChange(null);
      setModalPos(null);
    } else {
      setActiveChange(change);
      setModalPos({
        x: rect.left + rect.width / 2,
        y: rect.bottom + 8,
      });
    }
  };

  const segments = result ? buildSegments(result.rewrite, result.changes) : [];
  const hasContent = input.trim().length >= MIN_LENGTH;

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

  const wordCount = input.trim().split(/\s+/).filter(Boolean).length;
  const topBorder = "1px solid " + theme.border;
  const panelBorder = "1px solid " + theme.border;
  const toggleBtnBorder = "1px solid " + theme.toggleBorder;
  const copyBtnBorder = "1px solid " + theme.border;
  const modalOuterBorder = "1px solid " + theme.modalBorder;
  const accentBorderBottom = "1.5px solid " + accent;
  const mutedBorderBottom = "1.5px solid " + theme.textMuted;
  const placeholderColor = d ? "#382e24" : "#c8bfb4";
  const scrollThumbColor = d ? "#2a2418" : "#d0c8bc";

  const cssText = [
    "@keyframes pulse { 0%,100%{opacity:0.4} 50%{opacity:1} }",
    "@keyframes fadeIn { from{opacity:0} to{opacity:1} }",
    "@keyframes slideUp { from{opacity:0;transform:translateY(10px)} to{opacity:1;transform:translateY(0)} }",
    "textarea::placeholder { font-style: italic; color: " + placeholderColor + "; }",
    "* { box-sizing: border-box; }",
    "::-webkit-scrollbar { width: 3px; }",
    "::-webkit-scrollbar-thumb { background: " + scrollThumbColor + "; }",
  ].join("\n");

  return (
    <div style={{
      minHeight: "100vh",
      background: theme.bg,
      fontFamily: "Palatino, 'Palatino Linotype', serif",
      display: "flex",
      flexDirection: "column",
      transition: "background 0.25s",
    }}>
      {/* Top bar */}
      <div style={{
        padding: "20px 40px",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        borderBottom: topBorder,
      }}>
        <div style={{ display: "flex", alignItems: "baseline", gap: 10 }}>
          <span style={{
            fontSize: 11,
            letterSpacing: "0.35em",
            textTransform: "uppercase" as const,
            color: theme.textMuted,
          }}>Lexis</span>
          <span style={{
            width: 28, height: 1,
            background: theme.border,
            display: "inline-block",
            verticalAlign: "middle",
          }} />
          <span style={{
            fontSize: 11,
            color: theme.textFaint,
            letterSpacing: "0.12em",
          }}>editorial companion</span>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 20 }}>
          <div style={{
            display: "flex",
            alignItems: "center",
            gap: 8,
            opacity: hasContent || streaming ? 1 : 0,
            transition: "opacity 0.3s",
          }}>
            {streaming ? (
              <>
                {[0, 1, 2].map((i) => (
                  <div
                    key={i}
                    style={{
                      width: 5, height: 5, borderRadius: "50%",
                      background: accent,
                      animation: "pulse 1.2s ease-in-out infinite",
                      animationDelay: (i * 180) + "ms",
                    }}
                  />
                ))}
                <span style={{
                  fontSize: 10,
                  letterSpacing: "0.2em",
                  color: theme.textMuted,
                  textTransform: "uppercase" as const,
                  marginLeft: 4,
                }}>rewriting</span>
              </>
            ) : result ? (
              <>
                <div style={{
                  width: 6, height: 6, borderRadius: "50%",
                  background: "#7aaa7a",
                  boxShadow: "0 0 8px #7aaa7a80",
                }} />
                <span style={{
                  fontSize: 10,
                  letterSpacing: "0.2em",
                  color: theme.textMuted,
                  textTransform: "uppercase" as const,
                }}>
                  {result.changes?.length} change{result.changes?.length !== 1 ? "s" : ""}
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
              textTransform: "uppercase" as const,
              cursor: "pointer",
              fontFamily: "inherit",
              transition: "all 0.2s",
            }}
            onMouseEnter={e => e.currentTarget.style.color = theme.text}
            onMouseLeave={e => e.currentTarget.style.color = theme.textMuted}
          >{dark ? "Light" : "Dark"}</button>
        </div>
      </div>

      {/* Panels */}
      <div style={{
        flex: 1,
        display: "grid",
        gridTemplateColumns: "1fr 1fr",
        minHeight: "calc(100vh - 65px)",
      }}>
        {/* Left panel */}
        <div style={{
          padding: 48,
          borderRight: panelBorder,
          display: "flex",
          flexDirection: "column",
          position: "relative",
        }}>
          <div style={{
            fontSize: 10,
            letterSpacing: "0.3em",
            textTransform: "uppercase" as const,
            color: theme.label,
            marginBottom: 24,
          }}>Your text</div>
          <textarea
            autoFocus
            value={input}
            onChange={e => setInput(e.target.value)}
            placeholder={"Begin writing\u2026"}
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
          {/* Style directive config */}
          <div style={{
            borderTop: panelBorder,
            marginTop: 16,
            paddingTop: 12,
          }}>
            <button
              onClick={() => setStyleOpen(!styleOpen)}
              style={{
                background: "none",
                border: "none",
                color: styleDirective ? accent : theme.textMuted,
                fontSize: 10,
                letterSpacing: "0.2em",
                textTransform: "uppercase" as const,
                cursor: "pointer",
                fontFamily: "inherit",
                padding: 0,
                display: "flex",
                alignItems: "center",
                gap: 6,
              }}
              onMouseEnter={e => e.currentTarget.style.color = theme.text}
              onMouseLeave={e => e.currentTarget.style.color = styleDirective ? accent : theme.textMuted}
            >
              <span style={{
                display: "inline-block",
                transform: styleOpen ? "rotate(90deg)" : "rotate(0deg)",
                transition: "transform 0.15s",
                fontSize: 8,
              }}>&#9654;</span>
              Style directive{styleDirective ? " \u2022" : ""}
            </button>
            {styleOpen && (
              <div style={{ marginTop: 10, animation: "fadeIn 0.2s" }}>
                <textarea
                  value={styleDirective}
                  onChange={e => setStyleDirective(e.target.value)}
                  placeholder={"e.g. Write in the style of Cormac McCarthy \u2014 sparse punctuation, biblical cadence, visceral imagery\u2026"}
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
                  onFocus={e => e.currentTarget.style.borderColor = accent}
                  onBlur={e => e.currentTarget.style.borderColor = theme.toggleBorder}
                />
                {styleDirective && (
                  <button
                    onClick={() => setStyleDirective("")}
                    style={{
                      background: "none",
                      border: "none",
                      color: theme.textMuted,
                      fontSize: 9,
                      letterSpacing: "0.15em",
                      textTransform: "uppercase" as const,
                      cursor: "pointer",
                      fontFamily: "inherit",
                      padding: "6px 0 0 0",
                    }}
                    onMouseEnter={e => e.currentTarget.style.color = theme.text}
                    onMouseLeave={e => e.currentTarget.style.color = theme.textMuted}
                  >Clear</button>
                )}
              </div>
            )}
          </div>

          <div style={{
            fontSize: 10,
            color: theme.textFaint,
            letterSpacing: "0.15em",
            marginTop: 12,
          }}>
            {input.length > 0 ? wordCount + " words" : ""}
          </div>
        </div>

        {/* Right panel */}
        <div style={{
          padding: 48,
          display: "flex",
          flexDirection: "column",
          position: "relative",
          background: theme.bgR,
          transition: "background 0.25s",
          overflowY: "auto",
        }}>
          <div style={{
            fontSize: 10,
            letterSpacing: "0.3em",
            textTransform: "uppercase" as const,
            color: theme.label,
            marginBottom: 24,
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
          }}>
            <span>Alternate</span>
            <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
              {result && !streaming && (
                <span style={{
                  fontSize: 9,
                  color: theme.textFaint,
                  letterSpacing: "0.1em",
                  fontStyle: "italic",
                }}>click highlighted words</span>
              )}
              {result && !streaming && (
                <button
                  onClick={() => analyze(input.trim(), true)}
                  style={{
                    background: "none",
                    border: copyBtnBorder,
                    color: theme.textMuted,
                    padding: "4px 14px",
                    fontSize: 9,
                    letterSpacing: "0.2em",
                    textTransform: "uppercase" as const,
                    cursor: "pointer",
                    fontFamily: "inherit",
                    transition: "all 0.15s",
                  }}
                  onMouseEnter={e => { e.currentTarget.style.borderColor = theme.textMuted; e.currentTarget.style.color = theme.text; }}
                  onMouseLeave={e => { e.currentTarget.style.borderColor = theme.border; e.currentTarget.style.color = theme.textMuted; }}
                >Regenerate</button>
              )}
              {result && (
                <button
                  onClick={copyRewrite}
                  style={{
                    background: "none",
                    border: copyBtnBorder,
                    color: copied ? "#7aaa7a" : theme.textMuted,
                    padding: "4px 14px",
                    fontSize: 9,
                    letterSpacing: "0.2em",
                    textTransform: "uppercase" as const,
                    cursor: "pointer",
                    fontFamily: "inherit",
                    transition: "all 0.15s",
                  }}
                  onMouseEnter={e => { e.currentTarget.style.borderColor = theme.textMuted; e.currentTarget.style.color = theme.text; }}
                  onMouseLeave={e => { e.currentTarget.style.borderColor = theme.border; e.currentTarget.style.color = copied ? "#7aaa7a" : theme.textMuted; }}
                >{copied ? "Copied" : "Copy"}</button>
              )}
            </div>
          </div>

          {/* Empty state */}
          {!hasContent && !streaming && (
            <div style={{
              fontSize: 19,
              lineHeight: 1.85,
              color: theme.textFaint,
              fontStyle: "italic",
            }}>
              A rewritten version will appear here...
            </div>
          )}

          {/* Loading state */}
          {hasContent && streaming && !result && (
            <div style={{
              fontSize: 19,
              lineHeight: 1.85,
              color: theme.textFaint,
              fontStyle: "italic",
            }}>
              <span style={{ animation: "pulse 1.2s ease-in-out infinite" }}>Rewriting...</span>
            </div>
          )}

          {/* Annotated rewrite */}
          {result && (
            <div style={{
              fontSize: 19,
              lineHeight: 1.85,
              color: theme.text,
              fontFamily: "Palatino, serif",
              whiteSpace: "pre-wrap",
              wordBreak: "break-word",
              marginBottom: 32,
            }}>
              {segments.map((seg, i) => {
                if (seg.type === "plain") {
                  return <span key={i} style={{ color: theme.unchanged }}>{seg.text}</span>;
                }
                const isActive = activeChange?.start === seg.change?.start;
                const isToggled = toggledChanges.has(seg.change!.start);
                return (
                  <span
                    key={i}
                    onClick={(e) => handleHighlightClick(seg.change!, e)}
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
                    onMouseEnter={e => { e.currentTarget.style.background = theme.hoverBg; }}
                    onMouseLeave={e => { e.currentTarget.style.background = isActive ? theme.hoverBg : "transparent"; }}
                  >
                    {isToggled ? seg.change!.original : seg.text}
                  </span>
                );
              })}
            </div>
          )}

          {/* Changelog pills */}
          {result?.changes && result.changes.length > 0 && (
            <div style={{ marginTop: "auto", paddingTop: 24, borderTop: topBorder }}>
              <div style={{
                fontSize: 10,
                letterSpacing: "0.3em",
                textTransform: "uppercase" as const,
                color: theme.label,
                marginBottom: 14,
              }}>Changelog</div>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 7 }}>
                {result.changes.map((change, i) => {
                  const isToggled = toggledChanges.has(change.start);
                  const pillBorderVal = "1px solid " + (isToggled ? theme.textMuted : theme.pillBorder);
                  return (
                    <div
                      key={i}
                      onClick={() => toggleChange(change)}
                      style={{
                        background: isToggled ? (d ? "#1a1a14" : "#e8e4dc") : theme.pillBg,
                        border: pillBorderVal,
                        padding: "8px 14px",
                        cursor: "pointer",
                        display: "flex",
                        alignItems: "center",
                        gap: 8,
                        transition: "border-color 0.12s, background 0.12s",
                        fontSize: 14,
                        fontFamily: "Palatino, serif",
                      }}
                      onMouseEnter={e => e.currentTarget.style.borderColor = accent}
                      onMouseLeave={e => e.currentTarget.style.borderColor = isToggled ? theme.textMuted : theme.pillBorder}
                    >
                      <span style={{
                        fontStyle: "italic",
                        color: isToggled ? theme.accentText : theme.textMuted,
                      }}>
                        {isToggled ? change.original : change.original}
                      </span>
                      <span style={{ color: theme.textFaint, fontSize: 12 }}>
                        {isToggled ? "\u2190" : "\u2192"}
                      </span>
                      <span style={{
                        fontStyle: "italic",
                        color: isToggled ? theme.textMuted : theme.accentText,
                      }}>
                        {isToggled ? change.replacement : change.replacement}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Tone note */}
          {result?.analysis?.notes && (
            <div style={{
              fontSize: 12,
              fontStyle: "italic",
              color: theme.textMuted,
              marginTop: 18,
              lineHeight: 1.6,
            }}>
              <span style={{ color: accent, marginRight: 4 }}>&#10022;</span>
              {result.analysis.notes}
            </div>
          )}

          {/* Bottom legend */}
          {result && (
            <div style={{
              position: "absolute",
              bottom: 32,
              left: 48,
              display: "flex",
              alignItems: "center",
              gap: 16,
              fontSize: 10,
              color: theme.textFaint,
              letterSpacing: "0.12em",
            }}>
              <span style={{ display: "flex", alignItems: "center", gap: 5 }}>
                <span style={{ width: 16, height: 1.5, background: accent, display: "inline-block" }} />
                changed
              </span>
              <span style={{ color: theme.border }}>&middot;</span>
              <span>unchanged</span>
            </div>
          )}

          {/* Error */}
          {error && (
            <div style={{ color: "#aa5050", fontSize: 13, marginTop: 16, fontStyle: "italic" }}>
              {error}
            </div>
          )}
        </div>
      </div>

      {/* Floating Modal */}
      {activeChange && modalPos && (
        <div
          ref={modalRef}
          style={{
            position: "fixed",
            zIndex: 50,
            width: 380,
            maxWidth: "92vw",
            background: theme.modalBg,
            border: modalOuterBorder,
            boxShadow: d ? "0 24px 60px rgba(0,0,0,0.6)" : "0 16px 48px rgba(0,0,0,0.12)",
            left: Math.max(16, Math.min(modalPos.x - 190, window.innerWidth - 400)),
            top: modalPos.y,
            animation: "slideUp 0.18s ease",
          }}
        >
          <div style={{ padding: "24px 28px 20px", position: "relative" }}>
            <button
              onClick={() => { setActiveChange(null); setModalPos(null); }}
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
              onMouseEnter={e => e.currentTarget.style.color = d ? "#c8a87a" : "#6a5a4a"}
              onMouseLeave={e => e.currentTarget.style.color = d ? "#5a4e3a" : "#b0a090"}
            >&times;</button>

            <div style={{
              fontSize: 10,
              letterSpacing: "0.28em",
              textTransform: "uppercase" as const,
              color: d ? "#5a4e3a" : "#b0a090",
              marginBottom: 10,
            }}>Replaced</div>

            <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 14 }}>
              <span style={{
                fontFamily: "Palatino, serif",
                fontStyle: "italic",
                textDecoration: "line-through",
                color: d ? "#6a5a48" : "#a09080",
                fontSize: 18,
              }}>
                {activeChange.original}
              </span>
              <span style={{ color: theme.textFaint, fontSize: 14 }}>{"\u2192"}</span>
              <span style={{
                fontFamily: "Palatino, serif",
                fontStyle: "italic",
                fontWeight: 500,
                color: d ? "#c8a87a" : "#8a5a20",
                fontSize: 18,
              }}>
                {activeChange.replacement}
              </span>
            </div>

            <div style={{
              width: "100%",
              height: 1,
              background: theme.modalBorder,
              marginBottom: 12,
            }} />

            <div style={{
              fontSize: 13,
              lineHeight: 1.6,
              color: d ? "#9a8a78" : "#6a5a48",
            }}>
              {activeChange.reason}
            </div>
          </div>
        </div>
      )}

      <style dangerouslySetInnerHTML={{ __html: cssText }} />
    </div>
  );
}

export default App;
