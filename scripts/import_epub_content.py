#!/usr/bin/env python3
"""Import legally supplied EPUB files into the backend chapter tables.

Input manifest columns:
goodreads_id,epub_path,source_type,source_name,license

Only use this with public-domain, open-license, or admin-owned EPUB files.
The script does not search shadow libraries or scrape copyrighted books.
"""

from __future__ import annotations

import argparse
import csv
import html
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

import psycopg2


CONTAINER_NS = {"c": "urn:oasis:names:tc:opendocument:xmlns:container"}
OPF_NS = {"opf": "http://www.idpf.org/2007/opf"}
XHTML_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"[ \t\r\f\v]+")
BLANK_LINES_RE = re.compile(r"\n{3,}")


@dataclass
class ChapterData:
    order: int
    title: str
    content: str
    href: str


def clean_html(raw: str) -> str:
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", "", raw)
    text = re.sub(r"(?i)</(p|div|section|article|h[1-6]|li|br)>", "\n", text)
    text = XHTML_RE.sub("", text)
    text = html.unescape(text)
    text = SPACE_RE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    return BLANK_LINES_RE.sub("\n\n", text).strip()


def zip_read_text(epub: zipfile.ZipFile, name: str) -> str:
    return epub.read(name).decode("utf-8", errors="replace")


def resolve_href(base: str, href: str) -> str:
    base_path = Path(base).parent
    return str((base_path / href).as_posix())


def parse_epub(path: Path) -> list[ChapterData]:
    with zipfile.ZipFile(path) as epub:
        container = ET.fromstring(zip_read_text(epub, "META-INF/container.xml"))
        rootfile = container.find(".//c:rootfile", CONTAINER_NS)
        if rootfile is None:
            raise ValueError("EPUB container does not include a rootfile.")
        opf_path = rootfile.attrib["full-path"]
        opf = ET.fromstring(zip_read_text(epub, opf_path))

        manifest = {
            item.attrib["id"]: item.attrib
            for item in opf.findall(".//opf:manifest/opf:item", OPF_NS)
            if "id" in item.attrib and "href" in item.attrib
        }
        spine_items = [
            item.attrib["idref"]
            for item in opf.findall(".//opf:spine/opf:itemref", OPF_NS)
            if "idref" in item.attrib
        ]

        chapters: list[ChapterData] = []
        for order, idref in enumerate(spine_items, start=1):
            item = manifest.get(idref)
            if not item:
                continue
            media_type = item.get("media-type", "")
            if "html" not in media_type and "xhtml" not in media_type:
                continue
            href = resolve_href(opf_path, item["href"])
            content = clean_html(zip_read_text(epub, href))
            if not content:
                continue
            title = first_title(content, order)
            chapters.append(ChapterData(order=order, title=title, content=content, href=href))
        return chapters


def first_title(content: str, order: int) -> str:
    for line in content.splitlines():
        if line and len(line) <= 120:
            return line
    return f"Chapter {order}"


def connect_db(args: argparse.Namespace):
    return psycopg2.connect(
        host=args.db_host,
        port=args.db_port,
        dbname=args.db_name,
        user=args.db_user,
        password=args.db_password,
    )


def import_book(cursor, row: dict, dry_run: bool) -> tuple[str, int]:
    goodreads_id = row["goodreads_id"].strip()
    epub_path = Path(row["epub_path"]).expanduser()
    chapters = parse_epub(epub_path)
    if not chapters:
        return goodreads_id, 0

    if dry_run:
        return goodreads_id, len(chapters)

    cursor.execute("SELECT id, language FROM books WHERE goodreads_id = %s", (goodreads_id,))
    book = cursor.fetchone()
    if not book:
        raise ValueError(f"Book with Goodreads ID {goodreads_id} does not exist in the database.")
    book_id, language = book

    cursor.execute(
        """
        INSERT INTO book_content_bundles
            (book_id, source_type, source_name, original_file_name, language_code, created_at)
        VALUES (%s, %s, %s, %s, %s, NOW())
        RETURNING id
        """,
        (
            book_id,
            row.get("source_type") or "PUBLIC_DOMAIN_EPUB",
            row.get("source_name") or row.get("license") or "Legal EPUB import",
            epub_path.name,
            language,
        ),
    )
    bundle_id = cursor.fetchone()[0]

    cursor.execute("DELETE FROM chapters WHERE book_id = %s", (book_id,))
    for chapter in chapters:
        cursor.execute(
            """
            INSERT INTO chapters
                (book_id, chapter_order, title, content, content_bundle_id, source_type, source_href)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (
                book_id,
                chapter.order,
                chapter.title[:255],
                chapter.content,
                bundle_id,
                row.get("source_type") or "PUBLIC_DOMAIN_EPUB",
                chapter.href,
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
    return goodreads_id, len(chapters)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--db-host", default="localhost")
    parser.add_argument("--db-port", type=int, default=5432)
    parser.add_argument("--db-name", default="ebookreader")
    parser.add_argument("--db-user", default="postgres")
    parser.add_argument("--db-password", default="12345")
    args = parser.parse_args()

    rows = list(csv.DictReader(args.manifest.open(encoding="utf-8")))
    if args.dry_run:
        for row in rows:
            goodreads_id, count = import_book(None, row, dry_run=True)
            print(f"{goodreads_id}: {count} chapters")
        return

    with connect_db(args) as conn:
        with conn.cursor() as cursor:
            for row in rows:
                goodreads_id, count = import_book(cursor, row, dry_run=False)
                print(f"{goodreads_id}: imported {count} chapters")
        conn.commit()


if __name__ == "__main__":
    main()
