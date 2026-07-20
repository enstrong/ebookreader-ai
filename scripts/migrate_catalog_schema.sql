-- Run once before importing the broad Goodreads catalog into an existing local DB.
-- Hibernate will create the right shape for fresh databases, but it will not
-- reliably drop the old title unique constraint or widen chapter content.

ALTER TABLE IF EXISTS books DROP CONSTRAINT IF EXISTS books_title_key;
ALTER TABLE IF EXISTS books DROP CONSTRAINT IF EXISTS uk5mtto2jcmfrwfg0p1ui8mnweu;
ALTER TABLE IF EXISTS chapters ALTER COLUMN content TYPE TEXT;
ALTER TABLE IF EXISTS books ALTER COLUMN title TYPE varchar(1000);
ALTER TABLE IF EXISTS books ALTER COLUMN author TYPE varchar(1000);
ALTER TABLE IF EXISTS books ALTER COLUMN cover_url TYPE varchar(1000);
ALTER TABLE IF EXISTS books ALTER COLUMN external_url TYPE varchar(1000);
ALTER TABLE IF EXISTS books ALTER COLUMN language TYPE varchar(64);

DROP INDEX IF EXISTS ux_books_goodreads_id;
ALTER TABLE IF EXISTS books DROP CONSTRAINT IF EXISTS books_goodreads_id_key;

ALTER TABLE IF EXISTS books
    ADD CONSTRAINT books_goodreads_id_key UNIQUE (goodreads_id);
