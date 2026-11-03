
import html
import re
import sys
from html.parser import HTMLParser
from pathlib import Path
from typing import ClassVar

import pymupdf


class DocumentationParser(HTMLParser):
    """Extract the main documentation content from HTML."""

    IGNORED_TAGS: ClassVar[set[str]] = {
        "script",
        "style",
        "nav",
        "header",
        "footer",
        "noscript",
        "svg",
    }

    BLOCK_TAGS: ClassVar[set[str]] = {
        "h1",
        "h2",
        "h3",
        "h4",
        "p",
        "pre",
        "li",
        "tr",
    }

    MAIN_TAGS: ClassVar[set[str]] = {"main", "article"}
    MAIN_IDS: ClassVar[set[str]] = {
        "content",
        "main-content",
        "doc-content",
    }
    MAIN_CLASSES: ClassVar[set[str]] = {
        "content",
        "container",
        "main-content",
    }

    def __init__(self):
        super().__init__()
        self.parts = []
        self.ignored_depth = 0
        self.main_depth = 0
        self.main_found = False
        self.tag_stack = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)

        if tag in self.IGNORED_TAGS:
            self.ignored_depth += 1
            return

        if self.ignored_depth:
            return

        classes = set((attrs.get("class") or "").split())
        element_id = attrs.get("id", "")

        is_main = (
            tag in self.MAIN_TAGS
            or element_id in self.MAIN_IDS
            or bool(classes & self.MAIN_CLASSES)
        )

        if is_main:
            self.main_found = True
            self.main_depth += 1

        if self.main_found and tag in self.BLOCK_TAGS:
            self.parts.append("\n")

        self.tag_stack.append(tag)

    def handle_endtag(self, tag):
        if tag in self.IGNORED_TAGS and self.ignored_depth:
            self.ignored_depth -= 1
            return

        if self.ignored_depth:
            return

        if self.main_found and tag in self.BLOCK_TAGS:
            self.parts.append("\n")

        if tag in self.tag_stack:
            self.tag_stack.reverse()
            self.tag_stack.remove(tag)
            self.tag_stack.reverse()

        if tag in self.MAIN_TAGS and self.main_depth:
            self.main_depth -= 1

    def handle_data(self, data):
        if self.ignored_depth or not self.main_found:
            return

        if not data.strip():
            return

        if self.tag_stack and self.tag_stack[-1] == "pre":
            self.parts.append(data)
        else:
            self.parts.append(data.strip() + " ")

    def get_text(self):
        content = "".join(self.parts)
        content = re.sub(r"[ \t]+", " ", content)
        content = re.sub(r" *\n *", "\n", content)
        content = re.sub(r"\n{3,}", "\n\n", content)
        return content.strip()


def extract_html_text(html_path):
    parser = DocumentationParser()

    source = html_path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    parser.feed(source)
    parser.close()

    text = parser.get_text()

    if not text:
        raise ValueError("No main documentation content found.")

    return text


def text_to_html(text):
    blocks = []

    for block in text.split("\n"):
        block = block.strip()

        if not block:
            continue

        escaped = html.escape(block)
        blocks.append(f"<p>{escaped}</p>")

    body = "\n".join(blocks)

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{
                font-family: sans-serif;
                font-size: 10pt;
                line-height: 1.5;
                color: #222222;
            }}

            p {{
                margin-bottom: 8pt;
                white-space: pre-wrap;
            }}
        </style>
    </head>
    <body>
        {body}
    </body>
    </html>
    """


def convert_html_to_pdf(html_path, pdf_path):
    text = extract_html_text(html_path)

    if len(text) < 1000:
        raise ValueError(
            f"Insufficient content: {len(text)} characters."
        )

    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    story = pymupdf.Story(html=text_to_html(text))
    writer = pymupdf.DocumentWriter(str(pdf_path))

    page_width = 595
    page_height = 842
    margin = 45

    page_number = 0

    def rectfn(rect_num, filled):
        nonlocal page_number

        page_number += 1

        page_rect = pymupdf.Rect(
            0, 0, page_width, page_height
        )

        content_rect = pymupdf.Rect(
            margin,
            margin,
            page_width - margin,
            page_height - margin,
        )

        return page_rect, content_rect, None

    story.write(writer, rectfn)
    writer.close()

    # Validate the generated PDF.
    pdf = pymupdf.open(pdf_path)
    extracted = " ".join(page.get_text() for page in pdf)
    page_count = len(pdf)
    pdf.close()

    if len(extracted) < len(text) * 0.5:
        raise ValueError(
            "PDF validation failed: too much text is missing."
        )

    print(f"PDF created: {pdf_path}")
    print(f"HTML text extracted: {len(text)} characters")
    print(f"PDF text recovered: {len(extracted)} characters")
    print(f"Pages: {page_count}")


def main():
    if len(sys.argv) != 3:
        print(
            "Usage: python scripts/extract_html_to_pdf.py "
            "input.html output.pdf"
        )
        sys.exit(1)

    source = Path(sys.argv[1])
    destination = Path(sys.argv[2])

    if not source.exists():
        print(f"Source file not found: {source}")
        sys.exit(1)

    try:
        convert_html_to_pdf(source, destination)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"PDF generation failed: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
