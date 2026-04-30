-- Optional profile: high_qps_approximate — switch from HNSW to IVFFlat (run after ANALYZE with sufficient rows).
-- Replace lists with sqrt(rowcount) guidance at build time in production; 100 is a dev default.
DROP INDEX IF EXISTS chunks_embedding_hnsw;
DROP INDEX IF EXISTS chunks_embedding_ivfflat;
CREATE INDEX chunks_embedding_ivfflat ON chunks USING ivfflat (embedding vector_cosine_ops)
  WITH (lists = 100);
