# Recipe: IVFFlat-oriented profile

**Goal:** Align the **database index** with a profile that uses **IVFFlat** search knobs instead of HNSW.

## Warning

Default migrations create an **HNSW** index. If you set **`active_profile`** in `config/profiles.yaml` to an IVFFlat-oriented profile **without** the matching index, search can be wrong or fail.

## High-level flow

1. Read the comments in **[migrations/alternate/ivfflat_profile.sql](../../migrations/alternate/ivfflat_profile.sql)** and the “Switching index family” section in the main **[README](../../README.md)**.
2. Plan a maintenance window or empty DB if you are learning—IVFFlat setup often implies rebuilding or swapping indexes.
3. Run the alternate migration path when ready:

   ```bash
   uv run python scripts/migrate.py --alternate-ivfflat
   ```

   Only after you understand the SQL steps your deployment needs.

4. Set **`active_profile`** in `config/profiles.yaml` to the IVFFlat profile you prepared (e.g. `high_qps_approximate` if that matches your YAML).
5. Restart the API / container so config is reloaded.

## Verify

```bash
curl -s http://127.0.0.1:8000/config/active-profile
```

Check `index_family` and `ivfflat_probes` / HNSW fields match what you expect.
