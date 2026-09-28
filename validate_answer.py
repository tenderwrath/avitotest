import argparse
from pathlib import Path
import re
import pandas as pd

def validate_answer(path, queries, items):
    answer = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
    assert list(answer.columns) == ["query_id", "answer"], "Неверные колонки"
    assert len(answer) == len(queries), "Неверное число строк"
    assert answer.query_id.is_unique, "Повторяющиеся query_id"
    assert set(answer.query_id) == set(
        queries.query_id
    ), "Пропущенные или лишние запросы"
    assert answer.query_id.str.len().eq(16).all(), "query_id должен иметь 16 символов"
    valid_ids = set(items.item_id)
    lengths = []
    for row in answer.itertuples(index=False):
        ids = row.answer.split()
        assert len(ids) <= 50, f"Больше 50 кандидатов: {row.query_id}"
        assert len(ids) == len(set(ids)), f"Дубли в {row.query_id}"
        assert all(
            re.fullmatch(r"[0-9a-f]{16}", x) for x in ids
        ), "Неверный формат item_id"
        assert set(ids).issubset(
            valid_ids
        ), f"item_id отсутствует в корпусе: {row.query_id}"
        lengths.append(len(ids))
    result = {
        "rows": len(answer),
        "minimum_candidates": min(lengths),
        "maximum_candidates": max(lengths),
        "all_checks_passed": True,
    }
    print(result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("upload"))
    p.add_argument("--answer", type=Path, default=Path("answer.csv"))
    a = p.parse_args()
    validate_answer(
        a.answer,
        pd.read_parquet(a.data / "benchmark_queries.parquet"),
        pd.read_parquet(a.data / "benchmark_items.parquet", columns=["item_id"]),
    )
