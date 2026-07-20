#!/usr/bin/env python3
"""Import Russian public-domain texts and one LibriVox audiobook.

Text sources are limited to clearly public/free sources used for the diploma
demo:
- Project Gutenberg Russian-language texts.
- LibriVox-listed public-domain source text for "Белые ночи".

The White Nights audio tracks are LibriVox public-domain MP3 sections hosted on
the Internet Archive.
"""

from __future__ import annotations

import argparse
import html
import re
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import psycopg2
import requests


USER_AGENT = "ebookreader-diploma-import/1.0 (local development)"
SOURCE_GUTENBERG = "PROJECT_GUTENBERG_TEXT"
SOURCE_LIBRIVOX_TEXT = "LIBRIVOX_TEXT_SOURCE"
SOURCE_LIBRIVOX_AUDIO = "LIBRIVOX_AUDIO"
ASSET_ROOT = Path("backend/assets/audio/librivox/white_nights")


@dataclass(frozen=True)
class PublicDomainText:
    goodreads_id: str
    title: str
    author: str
    description: str
    cover_url: str
    external_url: str
    genres: tuple[str, ...]
    page_count: int
    text_url: str
    source_type: str = SOURCE_GUTENBERG
    source_name: str = "Project Gutenberg public domain text"


@dataclass(frozen=True)
class AudioSection:
    order: int
    title: str
    url: str
    duration_ms: int


TEXT_WORKS = (
    PublicDomainText(
        "pg-ru-19681",
        "Детство",
        "Leo Tolstoy",
        "Russian public-domain text of Tolstoy's first published novel.",
        "https://www.gutenberg.org/cache/epub/19681/pg19681.cover.medium.jpg",
        "https://www.gutenberg.org/ebooks/19681",
        ("fiction", "classics", "russian literature"),
        160,
        "https://www.gutenberg.org/files/19681/19681.txt",
    ),
    PublicDomainText(
        "pg-ru-16527",
        "1001 задача для умственного счета",
        "Sergei Rachinskii",
        "A public-domain Russian collection of mental arithmetic problems.",
        "https://www.gutenberg.org/cache/epub/16527/pg16527.cover.medium.jpg",
        "https://www.gutenberg.org/ebooks/16527",
        ("mathematics", "education", "public domain"),
        120,
        "https://www.gutenberg.org/ebooks/16527.txt.utf-8",
    ),
    PublicDomainText(
        "pg-ru-14741",
        "Духовные оды",
        "Gavriil Derzhavin",
        "Russian public-domain poetry by Gavriil Derzhavin.",
        "https://www.gutenberg.org/cache/epub/14741/pg14741.cover.medium.jpg",
        "https://www.gutenberg.org/ebooks/14741",
        ("poetry", "classics", "russian literature"),
        80,
        "https://www.gutenberg.org/ebooks/14741.txt.utf-8",
    ),
    PublicDomainText(
        "pg-ru-5316",
        "Красавице, которая нюхала табак",
        "Alexander Pushkin",
        "Russian public-domain poem by Alexander Pushkin.",
        "https://www.gutenberg.org/cache/epub/5316/pg5316.cover.medium.jpg",
        "https://www.gutenberg.org/ebooks/5316",
        ("poetry", "classics", "russian literature"),
        20,
        "https://www.gutenberg.org/ebooks/5316.txt.utf-8",
    ),
    PublicDomainText(
        "pg-ru-30774",
        "Московия в представлении иностранцев XVI-XVII в.",
        "P. N. Apostol",
        "A public-domain Russian historical text about foreign views of Muscovy.",
        "https://www.gutenberg.org/cache/epub/30774/pg30774.cover.medium.jpg",
        "https://www.gutenberg.org/ebooks/30774",
        ("history", "russian history", "public domain"),
        240,
        "https://www.gutenberg.org/ebooks/30774.txt.utf-8",
    ),
    PublicDomainText(
        "pg-ru-37196",
        "Женское международное движение: Сборник статей",
        "Various",
        "A public-domain Russian collection of articles on the international women's movement.",
        "https://www.gutenberg.org/cache/epub/37196/pg37196.cover.medium.jpg",
        "https://www.gutenberg.org/ebooks/37196",
        ("history", "essays", "public domain"),
        200,
        "https://www.gutenberg.org/ebooks/37196.txt.utf-8",
    ),
)

WHITE_NIGHTS = PublicDomainText(
    "pg-ru-21183",
    "Белые ночи",
    "Fyodor Dostoyevsky",
    "A synced Russian public-domain text and LibriVox recording of Dostoevsky's White Nights.",
    "https://www.gutenberg.org/cache/epub/21183/pg21183.cover.medium.jpg",
    "https://librivox.org/white-nights-by-fyodor-dostoevsky/",
    ("fiction", "classics", "russian literature"),
    96,
    "http://az.lib.ru/d/dostoewskij_f_m/text_0230.shtml",
    SOURCE_LIBRIVOX_TEXT,
    "LibriVox listed public-domain source text",
)

WHITE_NIGHTS_AUDIO = (
    AudioSection(
        1,
        "Ночь первая",
        "https://www.archive.org/download/white_nights_librivox/1-fm-dostoevsky-belye-nochi-night1_64kb.mp3",
        1_496_000,
    ),
    AudioSection(
        2,
        "Ночь вторая - часть 1",
        "https://www.archive.org/download/white_nights_librivox/2-fm-dostoevsky-belye-nochi-night2-part1_64kb.mp3",
        1_419_000,
    ),
    AudioSection(
        3,
        "Ночь вторая - часть 2",
        "https://www.archive.org/download/white_nights_librivox/3-fm-dostoevsky-belye-nochi-night2-part2_64kb.mp3",
        545_000,
    ),
    AudioSection(
        4,
        "История Настеньки",
        "https://www.archive.org/download/white_nights_librivox/4-fm-dostoevsky-belye-nochi-history_of_nastenka_64kb.mp3",
        1_247_000,
    ),
    AudioSection(
        5,
        "Ночь третья",
        "https://www.archive.org/download/white_nights_librivox/5-fm-dostoevsky-belye-nochi-night3_64kb.mp3",
        873_000,
    ),
    AudioSection(
        6,
        "Ночь четвертая",
        "https://www.archive.org/download/white_nights_librivox/6-fm-dostoevsky-belye-nochi-night4_64kb.mp3",
        1_155_000,
    ),
    AudioSection(
        7,
        "Утро",
        "https://www.archive.org/download/white_nights_librivox/7-fm-dostoevsky-belye-nochi-morning_64kb.mp3",
        352_000,
    ),
)


START_RE = re.compile(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK .*?\*\*\*", re.I | re.S)
END_RE = re.compile(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK .*?\*\*\*", re.I | re.S)
TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"[ \t\r\f\v]+")
BLANK_LINES_RE = re.compile(r"\n{3,}")
RUSSIAN_HEADING_RE = re.compile(
    r"(?im)^(?:"
    r"глава\s+[ivxlcdm\dа-яё]+.*"
    r"|часть\s+[ivxlcdm\dа-яё]+.*"
    r"|[IVXLCDM]{1,8}\.?"
    r"|[XVI]{1,6}"
    r")\s*$"
)


def request_text(url: str, encoding: str | None = None) -> str:
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=90)
    response.raise_for_status()
    if encoding:
        response.encoding = encoding
    elif response.encoding is None:
        response.encoding = response.apparent_encoding
    return response.text


def strip_gutenberg(text: str) -> str:
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


def strip_libru_html(text: str) -> str:
    text = html.unescape(TAG_RE.sub("\n", text))
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = SPACE_RE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    lines = [line for line in text.splitlines() if line.strip()]

    start = 0
    for index, line in enumerate(lines):
        if line.strip().lower() == "ночь первая":
            start = index
            break

    end = len(lines)
    for index, line in enumerate(lines[start:], start=start):
        if line.startswith("Комментарии") or line.startswith("Примечания"):
            end = index
            break
    return BLANK_LINES_RE.sub("\n\n", "\n".join(lines[start:end])).strip()


def split_by_size(text: str, chars: int = 24_000) -> list[tuple[str, str]]:
    chunks: list[tuple[str, str]] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chars)
        if end < len(text):
            break_at = text.rfind("\n\n", start, end)
            if break_at > start + chars // 2:
                end = break_at
        content = text[start:end].strip()
        if content:
            chunks.append((f"Часть {len(chunks) + 1}", content))
        start = max(end, start + 1)
    return chunks


def split_russian_text(text: str) -> list[tuple[str, str]]:
    matches = list(RUSSIAN_HEADING_RE.finditer(text))
    matches = [match for match in matches if match.start() > 200]
    if len(matches) < 2:
        return split_by_size(text)

    chapters: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        next_start = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        content = text[match.start() : next_start].strip()
        if len(content) >= 500:
            title = content.splitlines()[0].strip()[:255] or f"Часть {len(chapters) + 1}"
            chapters.append((title, content))
    return chapters or split_by_size(text)


def slice_between(text: str, start_heading: str, end_heading: str | None) -> str:
    start = text.lower().find(start_heading.lower())
    if start < 0:
        return ""
    end = len(text) if end_heading is None else text.lower().find(end_heading.lower(), start + len(start_heading))
    if end < 0:
        end = len(text)
    return text[start:end].strip()


def split_at_paragraph_ratio(text: str, ratios: tuple[float, ...]) -> list[str]:
    cuts: list[int] = []
    for ratio in ratios:
        target = int(len(text) * ratio)
        cut = text.rfind("\n", 0, target)
        cuts.append(cut if cut > 0 else target)

    parts: list[str] = []
    previous = 0
    for cut in cuts + [len(text)]:
        parts.append(text[previous:cut].strip())
        previous = cut
    return parts


def split_white_nights(text: str) -> list[tuple[str, str]]:
    first = slice_between(text, "Ночь первая", "Ночь вторая")
    second = slice_between(text, "Ночь вторая", "Ночь третья")
    third = slice_between(text, "Ночь третья", "Ночь четвертая")
    fourth = slice_between(text, "Ночь четвертая", "Утро")
    morning = slice_between(text, "Утро", None)
    second_parts = split_at_paragraph_ratio(second, (0.44, 0.61))
    return [
        ("Ночь первая", first),
        ("Ночь вторая - часть 1", second_parts[0]),
        ("Ночь вторая - часть 2", second_parts[1]),
        ("История Настеньки", second_parts[2]),
        ("Ночь третья", third),
        ("Ночь четвертая", fourth),
        ("Утро", morning),
    ]


def connect_db(args: argparse.Namespace):
    return psycopg2.connect(
        host=args.db_host,
        port=args.db_port,
        dbname=args.db_name,
        user=args.db_user,
        password=args.db_password,
    )


def get_or_create_genre_id(cursor, name: str) -> int:
    cursor.execute("SELECT id FROM genres WHERE name = %s", (name,))
    row = cursor.fetchone()
    if row:
        return row[0]
    cursor.execute("INSERT INTO genres (name) VALUES (%s) ON CONFLICT (name) DO NOTHING RETURNING id", (name,))
    row = cursor.fetchone()
    if row:
        return row[0]
    cursor.execute("SELECT id FROM genres WHERE name = %s", (name,))
    return cursor.fetchone()[0]


def upsert_book(cursor, work: PublicDomainText) -> int:
    cursor.execute(
        """
        INSERT INTO books
            (title, author, description, cover_url, goodreads_id, average_rating,
             ratings_count, review_count, external_url, language, page_count, availability)
        VALUES (%s, %s, %s, %s, %s, 4.0, 0, 0, %s, 'rus', %s, 'TEXT')
        ON CONFLICT (goodreads_id) DO UPDATE SET
            title = EXCLUDED.title,
            author = EXCLUDED.author,
            description = EXCLUDED.description,
            cover_url = EXCLUDED.cover_url,
            external_url = EXCLUDED.external_url,
            language = EXCLUDED.language,
            page_count = EXCLUDED.page_count
        RETURNING id
        """,
        (
            work.title,
            work.author,
            work.description,
            work.cover_url,
            work.goodreads_id,
            work.external_url,
            work.page_count,
        ),
    )
    book_id = cursor.fetchone()[0]
    for genre in work.genres:
        genre_id = get_or_create_genre_id(cursor, genre)
        cursor.execute(
            """
            INSERT INTO book_genres (book_id, genre_id)
            SELECT %s, %s
            WHERE NOT EXISTS (
                SELECT 1 FROM book_genres WHERE book_id = %s AND genre_id = %s
            )
            """,
            (book_id, genre_id, book_id, genre_id),
        )
    return book_id


def replace_text(cursor, book_id: int, work: PublicDomainText, chapters: list[tuple[str, str]]) -> None:
    cursor.execute("DELETE FROM chapters WHERE book_id = %s", (book_id,))
    cursor.execute("DELETE FROM book_content_bundles WHERE book_id = %s", (book_id,))
    cursor.execute(
        """
        INSERT INTO book_content_bundles
            (book_id, source_type, source_name, original_file_name, language_code, created_at)
        VALUES (%s, %s, %s, %s, 'rus', NOW())
        RETURNING id
        """,
        (book_id, work.source_type, work.source_name, Path(urlparse(work.text_url).path).name or work.goodreads_id),
    )
    bundle_id = cursor.fetchone()[0]
    for order, (title, content) in enumerate(chapters, start=1):
        cursor.execute(
            """
            INSERT INTO chapters
                (book_id, chapter_order, title, content, content_bundle_id, source_type, source_href)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (book_id, order, title[:255], content, bundle_id, work.source_type, work.text_url),
        )


def download_audio(section: AudioSection, dry_run: bool) -> str:
    filename = Path(urlparse(section.url).path).name
    target = ASSET_ROOT / filename
    audio_path = f"assets/audio/librivox/white_nights/{filename}"
    if dry_run or target.exists():
        return audio_path

    target.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(section.url, headers={"User-Agent": USER_AGENT}, stream=True, timeout=120) as response:
        response.raise_for_status()
        with target.open("wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 256):
                if chunk:
                    file.write(chunk)
    return audio_path


def replace_audio(cursor, book_id: int, dry_run: bool) -> None:
    cursor.execute("DELETE FROM audio_tracks WHERE book_id = %s", (book_id,))
    for section in WHITE_NIGHTS_AUDIO:
        audio_path = download_audio(section, dry_run)
        cursor.execute(
            """
            INSERT INTO audio_tracks
                (book_id, segment_order, title, audio_path, original_file_name,
                 content_type, duration_ms, created_at)
            VALUES (%s, %s, %s, %s, %s, 'audio/mpeg', %s, NOW())
            """,
            (
                book_id,
                section.order,
                section.title,
                audio_path,
                Path(audio_path).name,
                section.duration_ms,
            ),
        )


def import_work(cursor, work: PublicDomainText, dry_run: bool) -> tuple[str, int]:
    if work.source_type == SOURCE_LIBRIVOX_TEXT:
        raw = request_text(work.text_url, encoding="windows-1251")
        text = strip_libru_html(raw)
        chapters = split_white_nights(text)
    else:
        raw = request_text(work.text_url)
        text = strip_gutenberg(raw)
        chapters = split_russian_text(text)

    book_id = upsert_book(cursor, work)
    replace_text(cursor, book_id, work, chapters)
    if work.goodreads_id == WHITE_NIGHTS.goodreads_id:
        replace_audio(cursor, book_id, dry_run)
        cursor.execute("UPDATE books SET availability = 'SYNCED' WHERE id = %s", (book_id,))
    else:
        cursor.execute("UPDATE books SET availability = 'TEXT' WHERE id = %s", (book_id,))
    return work.title, len(chapters)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--delay-seconds", type=float, default=0.5)
    parser.add_argument("--db-host", default="localhost")
    parser.add_argument("--db-port", type=int, default=5432)
    parser.add_argument("--db-name", default="ebookreader")
    parser.add_argument("--db-user", default="postgres")
    parser.add_argument("--db-password", default="12345")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    works = (*TEXT_WORKS, WHITE_NIGHTS)
    imported = 0
    chapters = 0

    with connect_db(args) as conn:
        with conn.cursor() as cursor:
            for index, work in enumerate(works):
                if index > 0 and args.delay_seconds > 0:
                    time.sleep(args.delay_seconds)
                title, count = import_work(cursor, work, args.dry_run)
                imported += 1
                chapters += count
                print(f"{title}: {count} readable segments")
        if args.dry_run:
            conn.rollback()
        else:
            conn.commit()

    print(f"Imported {imported} Russian public-domain books, {chapters} readable segments.")


if __name__ == "__main__":
    main()
