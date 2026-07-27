from __future__ import annotations

import html
import re
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "docs" / "runtime-flow-walkthrough.html"
OUTPUT = ROOT / "docs" / "runtime-flow-walkthrough.pdf"
PAGE_WIDTH = 612
PAGE_HEIGHT = 792
MARGIN_LEFT = 54
MARGIN_TOP = 54
MARGIN_BOTTOM = 54
FONT_SIZE = 11
LEADING = 15
MAX_CHARS = 92


class Extractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.lines: list[str] = []
        self.current: list[str] = []
        self.pre = False
        self.block_tags = {"p", "div", "h1", "h2", "h3", "li", "ol", "ul"}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag == "pre":
            self.flush()
            self.pre = True
        elif tag == "li":
            self.flush()
            self.current.append("- ")
        elif tag in {"h1", "h2", "h3", "br"}:
            self.flush()

    def handle_endtag(self, tag: str) -> None:
        if tag == "pre":
            self.flush()
            self.pre = False
        elif tag in self.block_tags:
            self.flush()

    def handle_data(self, data: str) -> None:
        text = html.unescape(data)
        if self.pre:
            for line in text.splitlines():
                self.lines.append(line.rstrip())
            return

        text = re.sub(r"\s+", " ", text)
        if text.strip():
            self.current.append(text.strip() + " ")

    def flush(self) -> None:
        if self.current:
            line = "".join(self.current).strip()
            if line:
                self.lines.append(line)
            self.current = []
        elif self.lines and self.lines[-1] != "":
            self.lines.append("")


def collapse_blank_lines(lines: list[str]) -> list[str]:
    collapsed: list[str] = []
    for line in lines:
        if line == "" and collapsed and collapsed[-1] == "":
            continue
        collapsed.append(line)
    return collapsed


def wrap_lines(lines: list[str]) -> list[str]:
    wrapped: list[str] = []
    for line in lines:
        if not line:
            wrapped.append("")
            continue

        if line.startswith("- "):
            wrapped.extend(wrap_bullet(line))
            continue

        if len(line) <= MAX_CHARS:
            wrapped.append(line)
            continue

        words = line.split()
        current = ""
        for word in words:
            candidate = word if not current else f"{current} {word}"
            if len(candidate) > MAX_CHARS:
                wrapped.append(current)
                current = word
            else:
                current = candidate
        if current:
            wrapped.append(current)
    return wrapped


def wrap_bullet(line: str) -> list[str]:
    prefix = "- "
    continuation = "  "
    current = prefix
    output: list[str] = []
    for word in line[2:].split():
        candidate = word if current == prefix else f"{current} {word}"
        if len(candidate) > MAX_CHARS:
            output.append(current)
            current = f"{continuation}{word}"
        else:
            current = candidate
    output.append(current)
    return output


def paginate(lines: list[str]) -> list[list[str]]:
    usable_lines = int((PAGE_HEIGHT - MARGIN_TOP - MARGIN_BOTTOM) / LEADING)
    return [lines[i : i + usable_lines] for i in range(0, len(lines), usable_lines)]


def pdf_escape(text: str) -> str:
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def build_content_stream(page: list[str]) -> bytes:
    operators = ["BT", f"/F1 {FONT_SIZE} Tf"]
    y = PAGE_HEIGHT - MARGIN_TOP
    for line in page:
        operators.append(f"1 0 0 1 {MARGIN_LEFT} {y} Tm ({pdf_escape(line)}) Tj")
        y -= LEADING
    operators.append("ET")
    return "\n".join(operators).encode("latin-1", errors="replace")


def render_pdf(lines: list[str]) -> bytes:
    pages = paginate(lines)
    streams = [build_content_stream(page) for page in pages]

    objects: list[bytes] = []
    page_ids: list[int] = []
    content_ids: list[int] = []
    next_id = 3
    for _ in pages:
        page_ids.append(next_id)
        content_ids.append(next_id + 1)
        next_id += 2
    font_id = next_id

    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(f"<< /Type /Pages /Count {len(pages)} /Kids [{kids}] >>".encode())

    for _page_id, content_id, stream in zip(page_ids, content_ids, streams, strict=True):
        page_obj = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {PAGE_WIDTH} {PAGE_HEIGHT}] "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {content_id} 0 R >>"
        ).encode()
        objects.append(page_obj)
        content_obj = (
            b"<< /Length "
            + str(len(stream)).encode()
            + b" >>\nstream\n"
            + stream
            + b"\nendstream"
        )
        objects.append(content_obj)

    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for object_id, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{object_id} 0 obj\n".encode())
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")

    xref_start = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode())
    trailer = (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_start}\n%%EOF\n"
    )
    pdf.extend(trailer.encode())
    return bytes(pdf)


def extract_lines(source: Path) -> list[str]:
    parser = Extractor()
    parser.feed(source.read_text(encoding="utf-8"))
    parser.flush()
    return wrap_lines(collapse_blank_lines([line.rstrip() for line in parser.lines]))


def main() -> None:
    OUTPUT.write_bytes(render_pdf(extract_lines(SOURCE)))
    print(OUTPUT)


if __name__ == "__main__":
    main()
