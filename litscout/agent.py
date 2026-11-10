"""LitScout: plan -> search -> screen -> snowball -> extract -> synthesize -> verify.

Design rule: the model never invents a citation. The writer can only cite numbered
evidence cards built from real paper records, and every cited sentence is checked
against the cited papers' abstracts before it reaches the reader.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import json
import math
import re
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from . import sources as S
from .llm import LLM, parse_json

PLAN_SYS = """You plan literature searches. Return JSON:
{"subtopics":[{"name":str,"queries":[str,...]}...], "inclusion": str}
Give 3-5 subtopics that together cover the question, each with 2-3 short search queries (3-8 words,
the terms authors would use in titles). Include synonyms and method names. "inclusion" = one sentence
saying which papers count as relevant."""

SCREEN_SYS = """You screen papers for a literature review. Score each numbered paper:
3 = studies the question's task directly (any region), OR studies a closely related task in the question's
    exact setting, OR is a review/survey of the task. Reviews are valuable: score them 3 if on topic.
2 = closely related task or method that a reviewer would cite for context (e.g. a neighbouring basin,
    a related hazard model, a data source the task depends on).
1 = shares only a general technique or broad domain.
0 = off topic.
Do not penalise a paper just because it studies a different region if the method and task match.
Return JSON {"scores":[{"i":int,"score":int}]} covering every paper, no explanations. Judge only from the text given."""

EXTRACT_SYS = """Extract facts about ONE paper from its title and abstract only. Return JSON:
{"task":str,"method":str,"data":str,"region":str,"key_result":str,"evidence_quote":str}
- Use "" when the abstract does not say.
- key_result: the main finding in one sentence, with numbers if given.
- evidence_quote: copy, word for word, the single sentence from the abstract that supports key_result."""

SYNTH_SYS = """You write a literature review from numbered evidence cards. Rules:
- Use ONLY the cards. Cite every factual sentence with card numbers like [3] or [2][7].
- Never cite a number that is not a card. Never add facts that are not on the cited cards.
- Compare approaches: what each does, on what data, how well, and how they differ. Comparison and
  summary sentences must cite every card they compare, e.g. "Unlike [2], [5] uses graph networks".
- Then list research gaps: things the question asks about that the cards do NOT cover, or settings
  no card studies. Gaps may cite cards that show the closest existing work.
Return JSON {"sections":[{"heading":str,"text":str}...],"gaps":[str,...]}. One section per subtopic given."""

VERIFY_SYS = """Decide whether a sentence from a literature review is supported by the cited sources.
Return JSON {"verdict":"supported"|"partial"|"unsupported","reason":str}.
supported = every claim in the sentence is stated or directly implied by the sources.
partial = the gist is supported but a detail (number, region, method) is not.
unsupported = the sources do not back the main claim."""

CITE = re.compile(r"\[(\d+)\]")
SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")
STOP = {"the", "a", "an", "of", "in", "and", "or", "for", "to", "on", "with", "by", "is", "are", "we", "this", "that", "using", "based", "study", "paper", "from", "as", "at"}


@dataclass
class Card:
    n: int
    paper: S.Paper
    score: int
    facts: dict = field(default_factory=dict)


@dataclass
class Review:
    question: str
    plan: dict
    cards: list[Card]
    sections: list[dict]
    gaps: list[dict]
    removed: list[dict]
    stats: dict

    def to_dict(self) -> dict:
        return {
            "question": self.question, "plan": self.plan, "stats": self.stats,
            "cards": [{"n": c.n, "score": c.score, "facts": c.facts, **c.paper.to_dict()} for c in self.cards],
            "sections": self.sections, "gaps": self.gaps, "removed": self.removed,
        }


def tokens(t: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", t.lower()) if len(w) > 2 and w not in STOP}


class LitScout:
    def __init__(self, llm: LLM, search_fns: dict[str, Callable] | None = None, max_cards: int = 25,
                 snowball_seeds: int = 6, from_year: int | None = None, until_year: int | None = None,
                 log: Callable[[str], None] = print):
        self.llm = llm
        self.search = search_fns or {"openalex": S.search_openalex, "arxiv": S.search_arxiv, "s2": S.search_s2}
        self.max_cards, self.snowball_seeds = max_cards, snowball_seeds
        self.from_year, self.until_year = from_year, until_year
        self.log = log

    # ---- 1. plan
    def plan(self, question: str) -> dict:
        try:
            p = parse_json(self.llm.complete(PLAN_SYS, f"QUESTION: {question}", task="plan", json_mode=True, big=True))
        except (ValueError, json.JSONDecodeError):
            p = {}
        subs = [s for s in p.get("subtopics", []) if s.get("queries")] or [{"name": "Overview", "queries": [question]}]
        return {"subtopics": subs[:5], "inclusion": p.get("inclusion", "")}

    # ---- 2. search
    def gather(self, plan: dict, question: str) -> list[S.Paper]:
        queries = list(dict.fromkeys([question] + [q for s in plan["subtopics"] for q in s["queries"]]))

        def one_source(name: str, fn: Callable) -> list[S.Paper]:
            got, fails = [], 0
            for q in queries:
                if fails >= 3:  # source is rate-limiting us: stop asking it for this run
                    break
                try:
                    got += fn(q)
                except Exception as e:  # one source failing should not stop the review
                    fails += 1
                    self.log(f"  {name} failed for '{q}': {e.__class__.__name__}")
            return got

        # sources have separate rate limits, so query them in parallel
        with ThreadPoolExecutor(max_workers=len(self.search)) as ex:
            found = [p for part in ex.map(lambda kv: one_source(*kv), self.search.items()) for p in part]
        papers = [p for p in S.merge(found) if p.title and self._year_ok(p)]
        self.log(f"search: {len(queries)} queries -> {len(papers)} unique papers")
        return papers

    def _year_ok(self, p: S.Paper) -> bool:
        if p.year is None:
            return True
        return (self.from_year is None or p.year >= self.from_year) and (self.until_year is None or p.year <= self.until_year)

    # ---- 3. screen: cheap lexical ranking, then LLM scoring of the top slice
    def prerank(self, papers: list[S.Paper], question: str, plan: dict, keep: int) -> list[S.Paper]:
        if len(papers) <= keep:
            return papers
        docs = [f"{p.title}. {p.title}. {p.abstract}" for p in papers]
        target = question + " " + " ".join(q for s in plan["subtopics"] for q in s["queries"])
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True).fit(docs + [target])
        sim = linear_kernel(vec.transform([target]), vec.transform(docs)).ravel()
        prior = np.array([math.log1p(p.citations) for p in papers]) / 10.0  # gentle nudge toward established work
        order = np.argsort(-(sim + 0.05 * prior))
        return [papers[i] for i in order[:keep]]

    def screen(self, papers: list[S.Paper], question: str, inclusion: str) -> list[tuple[S.Paper, int]]:
        scored = []
        for i in range(0, len(papers), 10):
            batch = papers[i:i + 10]
            listing = "\n".join(f"[{k}] {p.title} ({p.year})\n{p.abstract[:450]}" for k, p in enumerate(batch))
            self.log(f"screening papers {i + 1}-{i + len(batch)} of {len(papers)}")
            try:
                # alternate two models so batches draw on separate per-minute token quotas
                task = "screen" if (i // 10) % 2 == 0 else "screen_alt"
                res = parse_json(self.llm.complete(SCREEN_SYS, f"QUESTION: {question}\nRELEVANT IF: {inclusion}\n\n{listing}", task=task, json_mode=True))
                got = {int(s["i"]): int(s["score"]) for s in res.get("scores", []) if str(s.get("i", "")).isdigit() or isinstance(s.get("i"), int)}
            except (ValueError, KeyError, json.JSONDecodeError, TypeError, requests.RequestException):
                got = {}
            scored += [(p, got.get(k, 0)) for k, p in enumerate(batch)]
        return scored

    # ---- 4. snowball: references and citing papers of the strongest seeds
    def snowball(self, kept: list[tuple[S.Paper, int]]) -> list[S.Paper]:
        seeds = sorted(kept, key=lambda t: (-t[1], -t[0].citations))[: self.snowball_seeds]
        more = []
        for p, _ in seeds:
            if not p.pid.startswith("openalex:"):
                hit = S.openalex_resolve(doi=p.doi, title=p.title) if (p.doi or p.title) else None
                if not hit:
                    continue
                p = hit
            wid = p.pid.split(":", 1)[1]
            try:
                more += S.openalex_works(p.refs[:40], f"ref-of:{wid}") if p.refs else []
                more += S.openalex_citing(wid, n=20)
            except RuntimeError as e:
                self.log(f"  snowball failed for {wid}: {e}")
        more = [m for m in S.merge(more) if self._year_ok(m)]
        self.log(f"snowball: {len(seeds)} seeds -> {len(more)} candidate papers")
        return more

    # ---- 5. extract
    def extract(self, paper: S.Paper) -> dict:
        try:
            f = parse_json(self.llm.complete(EXTRACT_SYS, f"TITLE: {paper.title}\nABSTRACT: {paper.abstract[:2500]}", task="extract", json_mode=True))
        except (ValueError, json.JSONDecodeError, requests.RequestException):
            f = {}
        f = {k: str(f.get(k, "") or "").strip() for k in ("task", "method", "data", "region", "key_result", "evidence_quote")}
        # keep the quote only if it really is in the abstract (copied, not paraphrased)
        if f["evidence_quote"] and _squash(f["evidence_quote"]) not in _squash(paper.abstract):
            f["evidence_quote"] = ""
        return f

    # ---- 6. synthesize
    def synthesize(self, question: str, plan: dict, cards: list[Card]) -> dict:
        listing = "\n\n".join(
            f"[{c.n}] {c.paper.title} ({c.paper.year}; {c.paper.venue or c.paper.source})\n"
            f"task: {c.facts.get('task')}; method: {c.facts.get('method')}; data: {c.facts.get('data')}; region: {c.facts.get('region')}\n"
            f"finding: {c.facts.get('key_result')}" for c in cards)
        subs = "; ".join(s["name"] for s in plan["subtopics"])
        try:
            return parse_json(self.llm.complete(SYNTH_SYS, f"QUESTION: {question}\nSUBTOPICS: {subs}\n\nCARDS:\n{listing}", task="synthesize", json_mode=True, big=True))
        except (ValueError, json.JSONDecodeError):
            return {"sections": [], "gaps": []}

    # ---- 7. verify every cited sentence
    def verify_sentence(self, sentence: str, cards_by_n: dict[int, Card]) -> dict:
        nums = [int(n) for n in CITE.findall(sentence)]
        if not nums:
            return {"verdict": "uncited"}
        bad = [n for n in nums if n not in cards_by_n]
        if bad:
            return {"verdict": "invalid_citation", "reason": f"cites {bad}, which are not real cards"}
        evidence = "\n\n".join(f"[{n}] {cards_by_n[n].paper.title}. {cards_by_n[n].paper.abstract[:1500]}" for n in nums)
        claim_words = tokens(CITE.sub("", sentence))
        overlap = len(claim_words & tokens(evidence)) / max(len(claim_words), 1)
        try:
            v = parse_json(self.llm.complete(VERIFY_SYS, f"SENTENCE: {sentence}\n\nSOURCES:\n{evidence}", task="verify", json_mode=True, big=True))
            verdict = v.get("verdict", "unsupported")
        except (ValueError, json.JSONDecodeError, requests.RequestException):
            v, verdict = {}, "unsupported"
        if verdict == "supported" and overlap < 0.2:  # cheap guard against an over-generous judge
            verdict = "partial"
        return {"verdict": verdict, "reason": v.get("reason", ""), "overlap": round(overlap, 2)}

    def verify_text(self, text: str, cards_by_n: dict[int, Card], keep_partial: bool = True) -> tuple[str, list[dict], dict]:
        kept, removed, tally = [], [], {}
        for s in [x.strip() for x in SENT.split(text) if x.strip()]:
            v = self.verify_sentence(s, cards_by_n)
            tally[v["verdict"]] = tally.get(v["verdict"], 0) + 1
            ok = v["verdict"] == "supported" or (keep_partial and v["verdict"] == "partial") or (v["verdict"] == "uncited" and len(tokens(s)) < 12)
            (kept if ok else removed).append(s if ok else {"sentence": s, **v})
        return " ".join(kept), removed, tally

    # ---- full run
    def run(self, question: str) -> Review:
        plan = self.plan(question)
        self.log(f"plan: {[s['name'] for s in plan['subtopics']]}")
        pool = self.gather(plan, question)
        self.log("screening the most relevant candidates")
        first = self.screen(self.prerank(pool, question, plan, keep=60), question, plan["inclusion"])
        kept = [(p, s) for p, s in first if s >= 2]
        seen = {p.key() for p in pool}
        extra = [p for p in self.snowball(kept) if p.key() not in seen]
        if extra:
            second = self.screen(self.prerank(extra, question, plan, keep=30), question, plan["inclusion"])
            kept += [(p, s) for p, s in second if s >= 2]
        kept = sorted(kept, key=lambda t: (-t[1], -t[0].citations))
        kept = [t for t in kept if t[0].abstract][: self.max_cards]
        self.log(f"screen: kept {len(kept)} papers")
        cards = []
        for i, (p, s) in enumerate(kept):
            self.log(f"reading paper {i + 1} of {len(kept)}: {p.title[:80]}")
            cards.append(Card(n=i + 1, paper=p, score=s, facts=self.extract(p)))
        self.log("writing the review from evidence cards")
        draft = self.synthesize(question, plan, cards)
        self.log("verifying every cited sentence against its sources")
        by_n = {c.n: c for c in cards}
        sections, removed, tally = [], [], {}
        for sec in draft.get("sections", []):
            if isinstance(sec, str):  # models sometimes return plain strings instead of {heading, text}
                sec = {"heading": "", "text": sec}
            text, rm, t = self.verify_text(sec.get("text", ""), by_n)
            removed += [{**r, "section": sec.get("heading", "")} for r in rm]
            for k, v in t.items():
                tally[k] = tally.get(k, 0) + v
            if text:
                sections.append({"heading": sec.get("heading", ""), "text": text})
        gaps = []
        for g in draft.get("gaps", []):
            g = str(g)
            v = self.verify_sentence(g, by_n) if CITE.search(g) else {"verdict": "uncited"}
            if v["verdict"] != "invalid_citation":
                gaps.append({"text": g, "verdict": v["verdict"]})
        used = sorted({int(n) for s in sections for n in CITE.findall(s["text"])})
        stats = {
            "papers_found": len(pool) + len(extra), "papers_screened": len(first) + (len(extra) and min(len(extra), 30)),
            "cards": len(cards), "cards_cited": len(used),
            "sentences": sum(tally.values()), "verdicts": tally,
            "supported_share": round(tally.get("supported", 0) / max(sum(v for k, v in tally.items() if k != "uncited"), 1), 3),
            "invalid_citations_removed": tally.get("invalid_citation", 0),
            "llm_calls": getattr(self.llm, "calls", 0) if isinstance(getattr(self.llm, "calls", 0), int) else len(self.llm.calls),
        }
        return Review(question, plan, cards, sections, gaps, removed, stats)


def _squash(t: str) -> str:
    return re.sub(r"[^a-z0-9]", "", t.lower())


def to_markdown(r: Review) -> str:
    lines = [f"# {r.question}", "", f"*{r.stats['cards']} papers reviewed; {r.stats['supported_share']:.0%} of cited sentences verified against their sources.*", ""]
    for s in r.sections:
        lines += [f"## {s['heading']}", "", s["text"], ""]
    if r.gaps:
        lines += ["## Research gaps", ""] + [f"- {g['text']}" for g in r.gaps] + [""]
    lines += ["## Comparison", "", "| # | Paper | Year | Task | Method | Data / region | Key result |", "|---|---|---|---|---|---|---|"]
    for c in r.cards:
        f = c.facts
        dr = "; ".join(x for x in (f.get("data"), f.get("region")) if x)
        lines.append(f"| {c.n} | [{c.paper.title}]({c.paper.url}) | {c.paper.year or ''} | {f.get('task','')} | {f.get('method','')} | {dr} | {f.get('key_result','')} |")
    lines += ["", "## References", ""]
    for c in r.cards:
        au = ", ".join(c.paper.authors[:3]) + (" et al." if len(c.paper.authors) > 3 else "")
        lines.append(f"[{c.n}] {au} ({c.paper.year}). {c.paper.title}. *{c.paper.venue or c.paper.source}*. {c.paper.url}")
    return "\n".join(lines) + "\n"
