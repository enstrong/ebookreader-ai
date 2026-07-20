#!/usr/bin/env python3
"""Seed a few public-domain readable chapters through the admin API.

The script is intentionally idempotent: it logs in as admin, finds target books,
checks existing chapters, and only posts missing chapter orders.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def load_api_base() -> str:
    env_path = os.path.join(ROOT, ".env")
    base = os.environ.get("API_BASE_URL")
    if not base and os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("API_BASE_URL="):
                    base = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    return (base or "http://localhost:8080").rstrip("/")


API_BASE = load_api_base()
API = f"{API_BASE}/api"
ADMIN_USER = os.environ.get("EBOOK_ADMIN_USER", "admin")
ADMIN_PASSWORD = os.environ.get("EBOOK_ADMIN_PASSWORD", "admin060606")


def request(method: str, path: str, payload: dict | None = None, token: str | None = None):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(f"{API}{path}", data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=20) as response:
            body = response.read().decode("utf-8")
            return json.loads(body) if body else None
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {path} failed: HTTP {exc.code} {body}") from exc


def login() -> str:
    data = request(
        "POST",
        "/auth/login",
        {"username": ADMIN_USER, "password": ADMIN_PASSWORD},
    )
    token = data.get("token") if isinstance(data, dict) else None
    if not token:
        raise RuntimeError("Admin login did not return a token")
    return token


BOOKS = [
    {
        "title": "The Complete Stories and Poems",
        "goodreads_id": "23919",
        "chapters": [
            {
                "chapterOrder": 1,
                "title": "The Tell-Tale Heart",
                "content": """TRUE! nervous, very, very dreadfully nervous I had been and am; but why will you say that I am mad? The disease had sharpened my senses, not destroyed, not dulled them. Above all was the sense of hearing acute. I heard all things in the heaven and in the earth. I heard many things in hell. How, then, am I mad?

Hearken! and observe how healthily, how calmly, I can tell you the whole story.

It is impossible to say how first the idea entered my brain; but once conceived, it haunted me day and night. Object there was none. Passion there was none. I loved the old man. He had never wronged me. He had never given me insult. For his gold I had no desire.

I think it was his eye! yes, it was this! He had the eye of a vulture, a pale blue eye, with a film over it. Whenever it fell upon me, my blood ran cold; and so by degrees, very gradually, I made up my mind to take the life of the old man, and thus rid myself of the eye forever.

Now this is the point. You fancy me mad. Madmen know nothing. But you should have seen me. You should have seen how wisely I proceeded, with what caution, with what foresight, with what dissimulation, I went to work! I was never kinder to the old man than during the whole week before I killed him.""",
            },
            {
                "chapterOrder": 2,
                "title": "The Cask of Amontillado",
                "content": """THE thousand injuries of Fortunato I had borne as I best could; but when he ventured upon insult, I vowed revenge. You, who so well know the nature of my soul, will not suppose that I gave utterance to a threat. At length I would be avenged; this was a point definitively settled.

I must not only punish, but punish with impunity. A wrong is unredressed when retribution overtakes its redresser. It is equally unredressed when the avenger fails to make himself felt as such to him who has done the wrong.

It must be understood that neither by word nor deed had I given Fortunato cause to doubt my good will. I continued, as was my wont, to smile in his face, and he did not perceive that my smile now was at the thought of his immolation.

He had a weak point, this Fortunato, although in other regards he was a man to be respected and even feared. He prided himself on his connoisseurship in wine. Few Italians have the true virtuoso spirit. For the most part their enthusiasm is adopted to suit the time and opportunity, to practise imposture upon the British and Austrian millionaires.""",
            },
            {
                "chapterOrder": 3,
                "title": "The Raven",
                "content": """Once upon a midnight dreary, while I pondered, weak and weary,
Over many a quaint and curious volume of forgotten lore,
While I nodded, nearly napping, suddenly there came a tapping,
As of someone gently rapping, rapping at my chamber door.
"It is some visitor," I muttered, "tapping at my chamber door;
Only this and nothing more."

Ah, distinctly I remember it was in the bleak December,
And each separate dying ember wrought its ghost upon the floor.
Eagerly I wished the morrow; vainly I had sought to borrow
From my books surcease of sorrow, sorrow for the lost Lenore,
For the rare and radiant maiden whom the angels name Lenore,
Nameless here forevermore.""",
            },
        ],
    },
    {
        "title": "Essential Tales and Poems",
        "goodreads_id": "32552",
        "chapters": [
            {
                "chapterOrder": 1,
                "title": "The Fall of the House of Usher",
                "content": """DURING the whole of a dull, dark, and soundless day in the autumn of the year, when the clouds hung oppressively low in the heavens, I had been passing alone, on horseback, through a singularly dreary tract of country.

At length I found myself, as the shades of the evening drew on, within view of the melancholy House of Usher. I know not how it was, but with the first glimpse of the building a sense of insufferable gloom pervaded my spirit.

I looked upon the scene before me, upon the mere house, and the simple landscape features of the domain, upon the bleak walls, upon the vacant eye-like windows, upon a few rank sedges, and upon a few white trunks of decayed trees, with an utter depression of soul.""",
            },
            {
                "chapterOrder": 2,
                "title": "The Masque of the Red Death",
                "content": """THE Red Death had long devastated the country. No pestilence had ever been so fatal, or so hideous. Blood was its avatar and its seal, the redness and the horror of blood.

There were sharp pains, and sudden dizziness, and then profuse bleeding at the pores, with dissolution. The scarlet stains upon the body and especially upon the face of the victim were the pest ban which shut him out from the aid and from the sympathy of his fellow-men.

But the Prince Prospero was happy and dauntless and sagacious. When his dominions were half depopulated, he summoned to his presence a thousand hale and light-hearted friends from among the knights and dames of his court, and with these retired to the deep seclusion of one of his castellated abbeys.""",
            },
            {
                "chapterOrder": 3,
                "title": "The Purloined Letter",
                "content": """AT Paris, just after dark one gusty evening in the autumn of 18--, I was enjoying the twofold luxury of meditation and a meerschaum, in company with my friend C. Auguste Dupin, in his little back library.

For one hour at least we had maintained a profound silence; while each, to any casual observer, might have seemed intently and exclusively occupied with the curling eddies of smoke that oppressed the atmosphere of the chamber.

For myself, however, I was mentally discussing certain topics which had formed matter for conversation between us at an earlier period of the evening. I mean the affair of the Rue Morgue, and the mystery attending the murder of Marie Roget.""",
            },
        ],
    },
    {
        "title": "The Raven and Other Poems",
        "goodreads_id": "269322",
        "chapters": [
            {
                "chapterOrder": 1,
                "title": "The Raven",
                "content": """Once upon a midnight dreary, while I pondered, weak and weary,
Over many a quaint and curious volume of forgotten lore,
While I nodded, nearly napping, suddenly there came a tapping,
As of someone gently rapping, rapping at my chamber door.
"It is some visitor," I muttered, "tapping at my chamber door;
Only this and nothing more."

Presently my soul grew stronger; hesitating then no longer,
"Sir," said I, "or Madam, truly your forgiveness I implore;
But the fact is I was napping, and so gently you came rapping,
And so faintly you came tapping, tapping at my chamber door,
That I scarce was sure I heard you"; here I opened wide the door;
Darkness there and nothing more.""",
            },
            {
                "chapterOrder": 2,
                "title": "Annabel Lee",
                "content": """It was many and many a year ago,
In a kingdom by the sea,
That a maiden there lived whom you may know
By the name of Annabel Lee;
And this maiden she lived with no other thought
Than to love and be loved by me.

I was a child and she was a child,
In this kingdom by the sea,
But we loved with a love that was more than love,
I and my Annabel Lee,
With a love that the winged seraphs of heaven
Coveted her and me.""",
            },
            {
                "chapterOrder": 3,
                "title": "Eldorado",
                "content": """Gaily bedight,
A gallant knight,
In sunshine and in shadow,
Had journeyed long,
Singing a song,
In search of Eldorado.

But he grew old,
This knight so bold,
And o'er his heart a shadow
Fell as he found
No spot of ground
That looked like Eldorado.

And, as his strength
Failed him at length,
He met a pilgrim shadow;
"Shadow," said he,
"Where can it be,
This land of Eldorado?"

"Over the Mountains
Of the Moon,
Down the Valley of the Shadow,
Ride, boldly ride,"
The shade replied,
"If you seek for Eldorado!" """,
            },
        ],
    },
]


def find_book(books: list[dict], target: dict) -> dict | None:
    for book in books:
        if str(book.get("goodreadsId") or book.get("goodreads_id") or "") == target["goodreads_id"]:
            return book
    for book in books:
        if (book.get("title") or "").strip() == target["title"]:
            return book
    return None


def main() -> int:
    print(f"Using API: {API}")
    token = login()
    books = request("GET", "/books", token=token)
    if not isinstance(books, list):
        raise RuntimeError("GET /books did not return a list")

    created = 0
    skipped = 0
    missing = []

    for target in BOOKS:
        book = find_book(books, target)
        if not book:
            missing.append(target["title"])
            continue
        book_id = book["id"]
        existing = request("GET", f"/admin/books/{book_id}/chapters", token=token) or []
        existing_orders = {int(chapter.get("chapterOrder", 0)) for chapter in existing}
        print(f"\n{target['title']} (id={book_id})")
        for chapter in target["chapters"]:
            if chapter["chapterOrder"] in existing_orders:
                skipped += 1
                print(f"  skip chapter {chapter['chapterOrder']}: already exists")
                continue
            request("POST", f"/admin/books/{book_id}/chapters", chapter, token=token)
            created += 1
            print(f"  created chapter {chapter['chapterOrder']}: {chapter['title']}")

    print(f"\nCreated: {created}; skipped: {skipped}; missing books: {len(missing)}")
    if missing:
        for title in missing:
            print(f"  missing: {title}")
    return 0 if not missing else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Seed failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
