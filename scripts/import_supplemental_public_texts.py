#!/usr/bin/env python3
"""Generate SQL that replaces placeholder supplemental texts with real content."""

from __future__ import annotations

import html
import re
import sys
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path


USER_AGENT = "ebookreader-diploma-import/1.0"


class ParagraphParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.in_entry = False
        self.depth = 0
        self.in_p = False
        self.parts: list[str] = []
        self.paragraphs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)
        classes = attrs_dict.get("class", "")
        if tag == "div" and "entry-content" in classes:
            self.in_entry = True
            self.depth = 1
            return
        if self.in_entry and tag == "div":
            self.depth += 1
        if self.in_entry and tag == "p":
            self.in_p = True
            self.parts = []

    def handle_endtag(self, tag: str) -> None:
        if self.in_entry and tag == "p" and self.in_p:
            text = normalize("".join(self.parts))
            if text:
                self.paragraphs.append(text)
            self.in_p = False
        if self.in_entry and tag == "div":
            self.depth -= 1
            if self.depth <= 0:
                self.in_entry = False

    def handle_data(self, data: str) -> None:
        if self.in_entry and self.in_p:
            self.parts.append(data)


def fetch_text(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    url = urllib.parse.urlunsplit((
        parsed.scheme,
        parsed.netloc,
        urllib.parse.quote(parsed.path),
        urllib.parse.quote(parsed.query, safe="=&"),
        urllib.parse.quote(parsed.fragment),
    ))
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as response:
        data = response.read()
        charset = response.headers.get_content_charset() or "utf-8"
    return data.decode(charset, errors="replace")


def normalize(text: str) -> str:
    text = html.unescape(text)
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_tags(text: str) -> str:
    text = re.sub(r"(?is)<script.*?</script>|<style.*?</style>", "", text)
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"(?i)</p>|</div>|</h[1-6]>", "\n", text)
    text = re.sub(r"(?s)<[^>]+>", "", text)
    return normalize(text)


def strip_gutenberg(text: str) -> str:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").splitlines()
    start = 0
    end = len(lines)
    for index, line in enumerate(lines):
        if "*** START" in line or "***START" in line:
            start = index + 1
            break
    for index, line in enumerate(lines):
        if "*** END" in line or "***END" in line:
            end = index
            break
    return normalize("\n".join(lines[start:end]))


def split_by_heading(text: str, pattern: str, fallback_chars: int = 18_000) -> list[tuple[str, str]]:
    matches = list(re.finditer(pattern, text, flags=re.IGNORECASE | re.MULTILINE))
    if len(matches) < 2:
        return split_by_size(text, fallback_chars)
    chapters: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        next_start = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        content = normalize(text[match.start():next_start])
        if len(content) >= 200:
            title = normalize(match.group(0))[:255]
            chapters.append((title, content))
    return chapters or split_by_size(text, fallback_chars)


def split_by_size(text: str, chars: int = 18_000) -> list[tuple[str, str]]:
    chapters: list[tuple[str, str]] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chars)
        if end < len(text):
            cut = text.rfind("\n\n", start, end)
            if cut > start + chars // 2:
                end = cut
        content = normalize(text[start:end])
        if content:
            chapters.append((f"Часть {len(chapters) + 1}", content))
        start = max(end, start + 1)
    return chapters


def clean_wikitext(text: str) -> str:
    text = re.sub(r"(?s)<ref.*?</ref>|<[^>]+>", "", text)
    text = re.sub(r"(?s)\{\{.*?\}\}", "", text)
    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"\[https?://[^\s\]]+\s*([^\]]*)\]", r"\1", text)
    text = text.replace("'''", "").replace("''", "")
    return normalize(text)


def split_wiki_sections(text: str, fallback_title: str) -> list[tuple[str, str]]:
    matches = list(re.finditer(r"^==+\s*(.+?)\s*==+\s*$", text, flags=re.MULTILINE))
    if len(matches) < 2:
        cleaned = clean_wikitext(text)
        return [(fallback_title, cleaned)] if cleaned else []
    sections: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        next_start = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        title = clean_wikitext(match.group(1))[:255]
        content = clean_wikitext(text[match.end():next_start])
        if len(content) >= 80 and not title.lower().startswith(("references", "дерек")):
            sections.append((title, content))
    return sections


def qara_sozder() -> list[tuple[str, str]]:
    index_html = fetch_text("https://laro.kz/abai-kara-sozder/")
    links = []
    for match in re.finditer(r'<a href="(https://laro\.kz/(?:abai-kara-sozder/)?abay[0-9]+(?:\.php|/))">([^<]+)</a>', index_html):
        href, title = match.groups()
        if href not in [link for link, _ in links]:
            links.append((href, f"{normalize(title)} сөз"))
    if len(links) != 45:
        raise RuntimeError(f"Expected 45 Qara Sozder links, got {len(links)}")
    chapters: list[tuple[str, str]] = []
    for url, title in links:
        parser = ParagraphParser()
        parser.feed(fetch_text(url))
        content = "\n\n".join(parser.paragraphs)
        if len(content) < 50:
            raise RuntimeError(f"Could not parse {url}")
        chapters.append((title, content))
    return chapters


def tolstoy_detstvo() -> list[tuple[str, str]]:
    chapters: list[tuple[str, str]] = []
    base = "https://rvb.ru/tolstoy/01text/vol_1/01text/0001-{index:02d}.htm"
    for index in range(1, 29):
        page = fetch_text(base.format(index=index))
        marker = re.search(r"(?is)Content goes here.*?-->", page)
        body = page[marker.end():] if marker else page
        body = re.split(r'(?is)<div class="chapter"\s+id="ch\d+">\s*<!--|<ul class="pager"', body, maxsplit=1)[0]
        title_match = re.search(r"(?is)<h2[^>]*>(.*?)</h2>", body)
        title = strip_tags(title_match.group(1)) if title_match else f"Глава {index}"
        body = re.sub(r"(?is)<div class=\"footnote\".*?</div>|<div class=\"page\".*?</div>|<ul class=\"pager\".*?</ul>", "", body)
        paragraphs = [strip_tags(part) for part in re.findall(r"(?is)<p[^>]*>(.*?)</p>", body)]
        content = "\n\n".join(part for part in paragraphs if part)
        if len(content) < 100:
            raise RuntimeError(f"Could not parse Tolstoy chapter {index}")
        chapters.append((title, content))
    return chapters


def russian_texts() -> dict[str, list[tuple[str, str]]]:
    notes = strip_tags(fetch_text("http://az.lib.ru/d/dostoewskij_f_m/text_0290.shtml"))
    start = notes.lower().find("записки из подполья")
    if start > 0:
        notes = notes[start:]
    ruslan_raw = fetch_text("https://ru.wikisource.org/w/index.php?title=Руслан_и_Людмила_(Пушкин)&action=raw")
    ruslan = clean_wikitext(ruslan_raw)
    derzhavin = strip_gutenberg(fetch_text("https://www.gutenberg.org/ebooks/14741.txt.utf-8"))
    pushkin_tobacco = strip_gutenberg(fetch_text("https://www.gutenberg.org/ebooks/5316.txt.utf-8"))
    return {
        "lv-210": tolstoy_detstvo(),
        "lv-559": split_by_heading(notes, r"^(?:[IVXLCDM]+|[0-9]+)\.?\s*$", 14_000),
        "lv-2145": split_by_heading(ruslan, r"^Песнь\s+\S+.*$", 8_000),
        "ru-pg-derzhavin-ody": split_by_size(derzhavin, 10_000),
        "ru-pg-pushkin-tabak": [("Красавице, которая нюхала табак", pushkin_tobacco)],
    }


def kazakh_texts() -> dict[str, list[tuple[str, str]]]:
    poems_raw = fetch_text("https://kk.wikibooks.org/w/index.php?title=Абай_өлеңдері&action=raw")
    altynsarin_raw = fetch_text("https://wikisource.org/w/index.php?title=Ыбырай_Алтынсарин&action=raw")
    qara = qara_sozder()
    return {
        "ap-kz-abai-qara-sozder": qara,
        "kz-abai-qara-sozder-readable": qara,
        "kz-abai-olenderi": split_wiki_sections(poems_raw, "Абай өлеңдері")[:80],
        "kz-altynsarin-tandamaly": split_wiki_sections(altynsarin_raw, "Ыбырай Алтынсарин")[:80],
    }


def sql_value(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def emit_replace(book_id: str, chapters: list[tuple[str, str]], source_type: str, source_name: str, source_href: str, language: str) -> list[str]:
    lines = [
        f"SELECT replace_public_text({sql_value(book_id)}, {sql_value(source_type)}, {sql_value(source_name)}, {sql_value(source_href)}, {sql_value(language)}, ARRAY["
    ]
    for index, (title, content) in enumerate(chapters):
        comma = "," if index + 1 < len(chapters) else ""
        lines.append(f"  ROW({sql_value(title)}, {sql_value(content)})::chapter_input{comma}")
    lines.append("]);")
    return lines


def main() -> int:
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/private/tmp/supplemental_public_texts.sql")
    kaz = kazakh_texts()
    rus = russian_texts()
    lines = [
        "BEGIN;",
        "CREATE TYPE chapter_input AS (title text, content text);",
        """
CREATE OR REPLACE FUNCTION replace_public_text(
    p_goodreads_id text,
    p_source_type text,
    p_source_name text,
    p_source_href text,
    p_language text,
    p_chapters chapter_input[]
) RETURNS void LANGUAGE plpgsql AS $$
DECLARE
    v_book_id bigint;
    v_bundle_id bigint;
    v_chapter chapter_input;
    v_order int := 1;
BEGIN
    SELECT id INTO v_book_id FROM books WHERE goodreads_id = p_goodreads_id;
    IF v_book_id IS NULL THEN
        RAISE NOTICE 'Book % not found', p_goodreads_id;
        RETURN;
    END IF;
    DELETE FROM chapters WHERE book_id = v_book_id;
    DELETE FROM book_content_bundles WHERE book_id = v_book_id;
    INSERT INTO book_content_bundles
        (book_id, source_type, source_name, original_file_name, language_code, created_at)
    VALUES (v_book_id, p_source_type, p_source_name, p_goodreads_id, p_language, NOW())
    RETURNING id INTO v_bundle_id;
    FOREACH v_chapter IN ARRAY p_chapters LOOP
        INSERT INTO chapters
            (book_id, chapter_order, title, content, content_bundle_id, source_type, source_href)
        VALUES (v_book_id, v_order, LEFT(v_chapter.title, 255), v_chapter.content, v_bundle_id, p_source_type, p_source_href);
        v_order := v_order + 1;
    END LOOP;
    UPDATE books
    SET availability = CASE
            WHEN EXISTS (SELECT 1 FROM audio_tracks WHERE book_id = v_book_id) THEN 'SYNCED'
            ELSE 'TEXT'
        END,
        average_rating = 4.0,
        ratings_count = 0,
        review_count = 0
    WHERE id = v_book_id;
END;
$$;
""",
        """
UPDATE books
SET average_rating = 4.0, ratings_count = 0, review_count = 0
WHERE goodreads_id LIKE 'lv-%'
   OR goodreads_id LIKE 'ap-kz-%'
   OR goodreads_id LIKE 'kz-%'
   OR goodreads_id LIKE 'ru-pg-%';
""",
        "DELETE FROM chapters WHERE book_id IN (SELECT id FROM books WHERE goodreads_id IN ('kz-altynsarin-tandamaly', 'kz-shakarim-tandamaly'));",
        "DELETE FROM book_content_bundles WHERE book_id IN (SELECT id FROM books WHERE goodreads_id IN ('kz-altynsarin-tandamaly', 'kz-shakarim-tandamaly'));",
        "UPDATE books SET availability = 'METADATA_ONLY' WHERE goodreads_id IN ('kz-altynsarin-tandamaly', 'kz-shakarim-tandamaly');",
    ]
    for goodreads_id, chapters in kaz.items():
        if chapters:
            lines.extend(emit_replace(goodreads_id, chapters, "PUBLIC_DOMAIN_WEB_TEXT", "Public-domain Kazakh text", "https://laro.kz/abai-kara-sozder/", "kaz"))
    for goodreads_id, chapters in rus.items():
        if chapters:
            lines.extend(emit_replace(goodreads_id, chapters, "PUBLIC_DOMAIN_WEB_TEXT", "Public-domain Russian text", "", "rus"))
    lines.extend([
        "DROP FUNCTION replace_public_text(text, text, text, text, text, chapter_input[]);",
        "DROP TYPE chapter_input;",
        "COMMIT;",
    ])
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {output}")
    for goodreads_id, chapters in {**kaz, **rus}.items():
        print(f"{goodreads_id}: {len(chapters)} chapters")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
