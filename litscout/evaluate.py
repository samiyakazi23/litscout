"""Benchmark: can LitScout find the papers experts cite?

For each published survey/review paper, the gold set is its reference list (as indexed by OpenAlex).
LitScout gets only the survey's topic (its title without words like "a review of"), searches with
the survey's publication year as the cut-off, and we measure recall of the gold set:

  baseline     one OpenAlex search for the topic, top N
  multi-query  LitScout's planned queries (OpenAlex + arXiv), top N after lexical ranking
  + snowball   multi-query plus references/citations of the strongest seeds, top N

All three are ranked by the same lexical scorer and cut to the same N, so differences come from
what the search *finds*, not from a bigger pool. The survey itself is excluded.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from . import sources as S
from .agent import LitScout

SURVEY_WORDS = re.compile(r"\b(a |an )?(comprehensive |systematic |critical |state[- ]of[- ]the[- ]art )?(review|survey|overview)s?( of| on)?\b[:\s]*", re.I)


def topic_from_title(title: str) -> str:
    t = SURVEY_WORDS.sub(" ", title)
    t = re.sub(r"\s+", " ", t).strip(" :-,.")
    return t


def gold_for(survey: S.Paper) -> set[str]:
    return {"openalex:" + r for r in survey.refs}


def recall(found: list[S.Paper], gold: set[str], n: int) -> float:
    ids = {p.pid for p in found[:n]}
    return round(len(ids & gold) / max(len(gold), 1), 3)


def to_oa(papers: list[S.Paper]) -> list[S.Paper]:
    """Map non-OpenAlex records (arXiv) to OpenAlex ids where possible, so they can match gold ids."""
    out = []
    for p in papers:
        if p.pid.startswith("openalex:"):
            out.append(p)
        elif p.doi:
            hit = S.openalex_resolve(doi=p.doi)
            out.append(hit or p)
        else:
            out.append(p)
    return out


def eval_survey(survey_id: str, agent: LitScout, ns=(50, 100, 200)) -> dict:
    survey = S.openalex_works([survey_id], "survey")[0]
    gold = gold_for(survey)
    topic = topic_from_title(survey.title)
    agent.until_year = survey.year
    plan = agent.plan(topic)
    keep = max(ns)

    def clean(ps):
        return [p for p in S.merge(ps) if p.pid != survey.pid and (p.year or 0) <= survey.year]

    # baseline: one search, first 4 pages (200 results) so it can fill N=200
    base = agent.prerank(clean(S.search_openalex(topic, n=50) + _more_openalex(topic, pages=4)), topic, plan, keep=keep)

    pool = clean(to_oa(agent.gather(plan, topic)))
    multi = agent.prerank(pool, topic, plan, keep=keep)

    seeds = [(p, 3) for p in multi[: agent.snowball_seeds * 2]]  # lexical seeds: no LLM screening in the benchmark
    snow = clean(agent.snowball(seeds))
    both = agent.prerank(clean(pool + snow), topic, plan, keep=keep)

    return {
        "survey": survey.title, "year": survey.year, "topic": topic, "gold_refs": len(gold),
        "queries": [q for s in plan["subtopics"] for q in s["queries"]],
        "recall": {f"@{n}": {"baseline": recall(base, gold, n), "multi_query": recall(multi, gold, n), "with_snowball": recall(both, gold, n)} for n in ns},
    }


def _more_openalex(q: str, pages: int = 4) -> list[S.Paper]:
    out = []
    for page in range(2, pages + 1):
        data = S._cached_get(S.OPENALEX, {"search": q, "per-page": 50, "page": page, "mailto": "litscout@users.noreply.github.com"})
        out += [S._oa_paper(w, q) for w in data.get("results", []) if w.get("display_name")]
    return out


def main() -> None:
    from .llm import GroqLLM

    bench = json.loads(Path("data/benchmark/surveys.json").read_text())
    import os
    fns = {"openalex": S.search_openalex}
    if os.getenv("BENCH_ARXIV") == "1":
        fns["arxiv"] = lambda q: S.search_arxiv(q, 12)
    agent = LitScout(GroqLLM(), search_fns=fns, snowball_seeds=6)
    results = []
    for s in bench:
        print("->", s["id"], s.get("note", ""), flush=True)
        try:
            results.append(eval_survey(s["id"], agent))
            print(json.dumps(results[-1]["recall"]), flush=True)
        except Exception as e:  # keep going; report what failed
            results.append({"survey_id": s["id"], "error": str(e)})
    ok = [r for r in results if "recall" in r]
    summary = {}
    for n in ("@50", "@100", "@200"):
        summary[n] = {k: round(sum(r["recall"][n][k] for r in ok) / max(len(ok), 1), 3) for k in ("baseline", "multi_query", "with_snowball")}
    out = {"n_surveys": len(ok), "mean_recall": summary, "surveys": results}
    Path("reports").mkdir(exist_ok=True)
    Path("reports/benchmark.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
