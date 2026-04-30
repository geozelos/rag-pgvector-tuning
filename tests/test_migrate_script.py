"""Migration helper logic."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_migrate_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "migrate.py"
    spec = importlib.util.spec_from_file_location("migrate_cli", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_split_sql_strips_comments_and_splits_statements() -> None:
    mod = _load_migrate_module()
    sql = """
    -- comment
    CREATE TABLE a (id int);
    CREATE TABLE b (id int);
    """
    parts = mod.split_sql(sql)
    assert len(parts) == 2
    assert "CREATE TABLE a" in parts[0]


def test_split_sql_empty_and_comments_only() -> None:
    mod = _load_migrate_module()
    assert mod.split_sql("-- only\n") == []
    assert mod.split_sql("") == []
