#!/usr/bin/env python3
"""Import public-domain Project Gutenberg texts as readable chapters.

This intentionally imports only a curated public-domain batch. Goodreads remains
metadata/recommendation data; readable text comes from legal Gutenberg files.
"""

from __future__ import annotations

import argparse
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

import psycopg2


USER_AGENT = "ebookreader-diploma-import/1.0 (local development)"
SOURCE_TYPE = "PROJECT_GUTENBERG_TEXT"
SOURCE_NAME = "Project Gutenberg public domain text"
MIN_CHAPTER_CHARS = 500
FALLBACK_CHARS = 30000


@dataclass(frozen=True)
class GutenbergWork:
    goodreads_id: str
    gutenberg_id: int
    title: str


DEFAULT_WORKS = (
    GutenbergWork("1885", 1342, "Pride and Prejudice"),
    GutenbergWork("18490", 84, "Frankenstein"),
    GutenbergWork("6324090", 11, "Alice's Adventures in Wonderland"),
    GutenbergWork("153747", 2701, "Moby-Dick; or, The Whale"),
    GutenbergWork("10210", 1260, "Jane Eyre"),
    GutenbergWork("11016", 1260, "Jane Eyre"),
    GutenbergWork("17245", 345, "Dracula"),
    GutenbergWork("656", 2600, "War and Peace"),
    GutenbergWork("7144", 2554, "Crime and Punishment"),
    GutenbergWork("4934", 28054, "The Brothers Karamazov"),
    GutenbergWork("5297", 174, "The Picture of Dorian Gray"),
    GutenbergWork("1953", 98, "A Tale of Two Cities"),
    GutenbergWork("2623", 1400, "Great Expectations"),
    GutenbergWork("2619", 1400, "Great Expectations"),
    GutenbergWork("7126", 1184, "The Count of Monte Cristo"),
    GutenbergWork("2956", 76, "Adventures of Huckleberry Finn"),
    GutenbergWork("24583", 74, "The Adventures of Tom Sawyer"),
    GutenbergWork("348914", 768, "Wuthering Heights"),
    GutenbergWork("507157", 768, "Wuthering Heights"),
    GutenbergWork("6252154", 514, "Little Women"),
    GutenbergWork("372311", 514, "Little Women"),
    GutenbergWork("1381", 1727, "The Odyssey"),
    GutenbergWork("22221", 2199, "The Iliad"),
    GutenbergWork("1661", 1661, "The Adventures of Sherlock Holmes"),
    GutenbergWork("834", 834, "The Memoirs of Sherlock Holmes"),
    GutenbergWork("108", 108, "The Return of Sherlock Holmes"),
    GutenbergWork("244", 244, "A Study in Scarlet"),
    GutenbergWork("2097", 2097, "The Sign of the Four"),
    GutenbergWork("2852", 2852, "The Hound of the Baskervilles"),
)


START_RE = re.compile(
    r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK .*?\*\*\*",
    re.IGNORECASE | re.DOTALL,
)
END_RE = re.compile(
    r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK .*?\*\*\*",
    re.IGNORECASE | re.DOTALL,
)
SPACE_RE = re.compile(r"[ \t\r\f\v]+")
BLANK_LINES_RE = re.compile(r"\n{3,}")
HEADING_RE = re.compile(
    r"(?im)^(?:"
    r"chapter\s+(?:[ivxlcdm]+|\d+|[a-z]+)\b[^\n]{0,100}"
    r"|book\s+(?:[ivxlcdm]+|\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b[^\n]{0,100}"
    r"|part\s+(?:[ivxlcdm]+|\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b[^\n]{0,100}"
    r"|[IVXLCDM]{1,8}\.?"
    r")\s*$"
)


def candidate_urls(gutenberg_id: int) -> list[str]:
    return [
        f"https://www.gutenberg.org/cache/epub/{gutenberg_id}/pg{gutenberg_id}.txt",
        f"https://www.gutenberg.org/cache/epub/{gutenberg_id}/pg{gutenberg_id}-0.txt",
        f"https://www.gutenberg.org/files/{gutenberg_id}/{gutenberg_id}-0.txt",
        f"https://www.gutenberg.org/files/{gutenberg_id}/{gutenberg_id}.txt",
    ]


def fetch_text(gutenberg_id: int) -> tuple[str, str]:
    request_headers = {"User-Agent": USER_AGENT}
    last_error: Exception | None = None
    for url in candidate_urls(gutenberg_id):
        request = urllib.request.Request(url, headers=request_headers)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                raw = response.read()
            return raw.decode("utf-8", errors="replace"), url
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
    raise RuntimeError(f"Could not download Project Gutenberg text {gutenberg_id}: {last_error}")


def strip_gutenberg_boilerplate(text: str) -> str:
    text = text.replace("\ufeff", "").replace("\r\n", "\n").replace("\r", "\n")
    start = START_RE.search(text)
    if start:
        text = text[start.end() :]
    end = END_RE.search(text)
    if end:
        text = text[: end.start()]
    text = SPACE_RE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    return BLANK_LINES_RE.sub("\n\n", text).strip()


def chapter_title(content: str, fallback: int) -> str:
    for line in content.splitlines():
        if MIN_CHAPTER_CHARS > len(line.strip()) > 0 and len(line.strip()) <= 180:
            return line.strip()
    return f"Chapter {fallback}"


def split_by_size(text: str) -> list[tuple[str, str]]:
    chunks: list[tuple[str, str]] = []
    start = 0
    order = 1
    while start < len(text):
        end = min(len(text), start + FALLBACK_CHARS)
        if end < len(text):
            paragraph_break = text.rfind("\n\n", start, end)
            if paragraph_break > start + FALLBACK_CHARS // 2:
                end = paragraph_break
        content = text[start:end].strip()
        if content:
            chunks.append((f"Part {order}", content))
            order += 1
        start = end
    return chunks


def split_chapters(text: str) -> list[tuple[str, str]]:
    matches = list(HEADING_RE.finditer(text))
    usable_matches = [match for match in matches if match.start() > 500]
    if len(usable_matches) < 2:
        return split_by_size(text)

    chapters: list[tuple[str, str]] = []
    intro = text[: usable_matches[0].start()].strip()
    if len(intro) >= MIN_CHAPTER_CHARS:
        chapters.append(("Front Matter", intro))

    for index, match in enumerate(usable_matches):
        next_start = usable_matches[index + 1].start() if index + 1 < len(usable_matches) else len(text)
        content = text[match.start() : next_start].strip()
        if len(content) < MIN_CHAPTER_CHARS:
            continue
        chapters.append((chapter_title(content, len(chapters) + 1), content))
    return chapters or split_by_size(text)


def connect_db(args: argparse.Namespace):
    return psycopg2.connect(
        host=args.db_host,
        port=args.db_port,
        dbname=args.db_name,
        user=args.db_user,
        password=args.db_password,
    )


def import_work(cursor, work: GutenbergWork, dry_run: bool) -> tuple[str, int, bool]:
    cursor.execute(
        "SELECT id, title, language FROM books WHERE goodreads_id = %s",
        (work.goodreads_id,),
    )
    row = cursor.fetchone()
    if not row:
        return work.goodreads_id, 0, False

    book_id, db_title, language = row
    raw_text, source_url = fetch_text(work.gutenberg_id)
    text = strip_gutenberg_boilerplate(raw_text)
    chapters = split_chapters(text)

    if dry_run:
        return work.goodreads_id, len(chapters), True

    cursor.execute("DELETE FROM chapters WHERE book_id = %s", (book_id,))
    cursor.execute(
        "DELETE FROM book_content_bundles WHERE book_id = %s AND source_type = %s",
        (book_id, SOURCE_TYPE),
    )
    cursor.execute(
        """
        INSERT INTO book_content_bundles
            (book_id, source_type, source_name, original_file_name, language_code, created_at)
        VALUES (%s, %s, %s, %s, %s, NOW())
        RETURNING id
        """,
        (
            book_id,
            SOURCE_TYPE,
            SOURCE_NAME,
            f"pg{work.gutenberg_id}.txt",
            language or "eng",
        ),
    )
    bundle_id = cursor.fetchone()[0]

    for order, (title, content) in enumerate(chapters, start=1):
        cursor.execute(
            """
            INSERT INTO chapters
                (book_id, chapter_order, title, content, content_bundle_id, source_type, source_href)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (
                book_id,
                order,
                title[:255] or f"Chapter {order}",
                content,
                bundle_id,
                SOURCE_TYPE,
                source_url,
            ),
        )

    cursor.execute(
        """
        UPDATE books
        SET availability = CASE
            WHEN EXISTS (SELECT 1 FROM audio_tracks WHERE book_id = %s) THEN 'SYNCED'
            ELSE 'TEXT'
        END
        WHERE id = %s
        """,
        (book_id, book_id),
    )
    print(f"{work.goodreads_id}: {db_title} -> {len(chapters)} chapters")
    return work.goodreads_id, len(chapters), True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--delay-seconds", type=float, default=1.0)
    parser.add_argument("--db-host", default="localhost")
    parser.add_argument("--db-port", type=int, default=5432)
    parser.add_argument("--db-name", default="ebookreader")
    parser.add_argument("--db-user", default="postgres")
    parser.add_argument("--db-password", default="12345")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    imported = 0
    skipped = 0
    chapters = 0

    with connect_db(args) as conn:
        with conn.cursor() as cursor:
            for index, work in enumerate(DEFAULT_WORKS):
                if index > 0 and args.delay_seconds > 0:
                    time.sleep(args.delay_seconds)
                try:
                    _, count, found = import_work(cursor, work, args.dry_run)
                except Exception as exc:
                    print(f"{work.goodreads_id}: failed ({exc})")
                    skipped += 1
                    continue
                if not found:
                    print(f"{work.goodreads_id}: skipped, no matching Goodreads row")
                    skipped += 1
                    continue
                imported += 1
                chapters += count
        if args.dry_run:
            conn.rollback()
        else:
            conn.commit()

    print(f"Imported {imported} readable books, {chapters} chapters; skipped {skipped}.")


if __name__ == "__main__":
    main()
