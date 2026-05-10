-- Optional JSON metadata per chunk (JSONB containment filters on retrieve).
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS metadata jsonb NOT NULL DEFAULT '{}'::jsonb;

CREATE INDEX IF NOT EXISTS chunks_metadata_gin_idx ON chunks USING gin (metadata jsonb_path_ops);
