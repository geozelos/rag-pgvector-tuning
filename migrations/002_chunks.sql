-- Baseline corpus table: btree filters + vector column (dimension MUST match embedding_model.embedding_dim config).
CREATE TABLE IF NOT EXISTS chunks (
  id bigserial PRIMARY KEY,
  tenant_id text NOT NULL,
  source_type text NOT NULL,
  doc_id text NOT NULL,
  chunk_index int NOT NULL,
  content text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  embedding vector(768) NOT NULL,
  UNIQUE (doc_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS chunks_tenant_id_idx ON chunks (tenant_id);
CREATE INDEX IF NOT EXISTS chunks_source_type_idx ON chunks (source_type);
CREATE INDEX IF NOT EXISTS chunks_doc_id_idx ON chunks (doc_id);
CREATE INDEX IF NOT EXISTS chunks_created_at_idx ON chunks (created_at DESC);
