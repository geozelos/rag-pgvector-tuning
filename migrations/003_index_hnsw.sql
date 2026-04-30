-- Default profile: small_corpus_balanced — HNSW with conservative build params.
DROP INDEX IF EXISTS chunks_embedding_ivfflat;
DROP INDEX IF EXISTS chunks_embedding_hnsw;
CREATE INDEX chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);
