"""Command line: python -m litscout.cli "your research question" [--until 2024] [--out reports/x]"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from .agent import LitScout, to_markdown
from .llm import GroqLLM


def slug(q: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", q.lower()).strip("-")[:60]


def main() -> None:
    ap = argparse.ArgumentParser(description="LitScout: verified literature reviews")
    ap.add_argument("question")
    ap.add_argument("--from-year", type=int)
    ap.add_argument("--until", type=int)
    ap.add_argument("--max-cards", type=int, default=25)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    llm = GroqLLM()
    t0 = time.time()
    agent = LitScout(llm, max_cards=a.max_cards, from_year=a.from_year, until_year=a.until)
    review = agent.run(a.question)
    review.stats["seconds"] = round(time.time() - t0)
    review.stats["tokens"] = llm.tokens
    out = Path(a.out or f"reports/reviews/{slug(a.question)}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "review.md").write_text(to_markdown(review), encoding="utf-8")
    (out / "review.json").write_text(json.dumps(review.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(review.stats, indent=1))
    print(f"-> {out}/review.md")


if __name__ == "__main__":
    main()
