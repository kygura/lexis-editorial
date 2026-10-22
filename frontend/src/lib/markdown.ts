export type BlockKind = "prose" | "heading" | "code" | "rule" | "empty";

const FENCE = /^\s*(```|~~~)/;
const HEADING = /^#{1,6}\s/;
const RULE = /^\s*([-*_])(\s*\1){2,}\s*$/;
const FRONT_MATTER = /^---\n[\s\S]*?\n---(\n|$)/;

// Splits Markdown into paragraph-sized blocks: blank lines separate blocks,
// fenced code stays whole, and headings and rules always stand alone.
export function splitMarkdown(markdown: string): string[] {
  const text = markdown.replace(/\r\n?/g, "\n").replace(FRONT_MATTER, "");
  const blocks: string[] = [];
  let current: string[] = [];
  let fence: string | null = null;

  const flush = () => {
    const block = current.join("\n").replace(/^\n+|\s+$/g, "");
    if (block) blocks.push(block);
    current = [];
  };

  for (const line of text.split("\n")) {
    if (fence) {
      current.push(line);
      if (line.trim().startsWith(fence)) {
        fence = null;
        flush();
      }
      continue;
    }

    const opening = line.match(FENCE);
    if (opening) {
      flush();
      fence = opening[1];
      current.push(line);
      continue;
    }

    if (HEADING.test(line) || RULE.test(line)) {
      flush();
      current.push(line);
      flush();
      continue;
    }

    if (!line.trim()) {
      flush();
      continue;
    }

    current.push(line);
  }

  flush();
  return blocks;
}

export function blockKind(text: string): BlockKind {
  if (!text.trim()) return "empty";
  if (FENCE.test(text)) return "code";
  if (HEADING.test(text)) return "heading";
  if (RULE.test(text)) return "rule";
  return "prose";
}

export function wordCount(text: string) {
  return text.trim().split(/\s+/).filter(Boolean).length;
}
