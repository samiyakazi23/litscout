"""Paper mode: turn a project repository into a draft research paper whose claims are checked.

    python -m litscout.paper samiyakazi23/floodwatch --review docs/reviews/brahmaputra-floods.json

1. read the repo (README, results files, module docstrings) without running any of its code
2. turn every number in the README and results files into a numbered fact [F#] tied to its file
3. profile the project (problem, data, method, contributions) and find related work with LitScout [C#]
4. draft each section; the writer may cite only facts and cards it was given
5. check: a number about the project must appear in the fact it cites (else flagged for the author);
   a claim about other work is checked against the cited abstracts (unsupported -> cut);
   a citation to a fact or card that does not exist -> cut
6. reviewer checklist, closest existing work per contribution, Markdown / LaTeX / BibTeX export
"""
from __future__ import annotations

import argparse
import io
import json
import re
import time
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import requests

from . import sources as S
from .agent import Card, LitScout, tokens
from .llm import GroqLLM, parse_json

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIRS = ("reports", "results", "outputs", "output", "eval", "evaluation", "metrics", "benchmark", "benchmarks")
SKIP_JSON = {"package.json", "package-lock.json", "tsconfig.json", "manifest.json", ".eslintrc.json"}
MAX_FILE = 200_000

FACT_CITE = re.compile(r"\[F(\d+)\]")
CARD_CITE = re.compile(r"\[C(\d+)\]")
ANY_CITE = re.compile(r"\[[FC]\d+\]")
SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")
NUM = re.compile(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?(?:\s?%)?")


# ---------------------------------------------------------------- 1. read the repo
@dataclass
class Repo:
    name: str
    files: dict[str, str]                     # path -> text (README, results, small docs)
    docstrings: dict[str, str] = field(default_factory=dict)
    tree: list[str] = field(default_factory=list)


def read_repo(src: str) -> Repo:
    """Read a GitHub repo ('owner/repo' or URL) or a local folder. Nothing from it is executed."""
    m = re.match(r"(?:https?://github\.com/)?([\w.-]+)/([\w.-]+?)(?:\.git)?/?$", src.strip())
    if Path(src).exists():
        root = Path(src)
        items = {str(p.relative_to(root)): p for p in root.rglob("*") if p.is_file() and ".git" not in p.parts}
        get = lambda k: items[k].read_bytes()  # noqa: E731
        sizes = {k: p.stat().st_size for k, p in items.items()}
        name = root.resolve().name
    elif m:
        owner, repo = m.group(1), m.group(2)
        r = requests.get(f"https://codeload.github.com/{owner}/{repo}/zip/HEAD", timeout=60)
        if r.status_code != 200:
            raise ValueError(f"Could not download github.com/{owner}/{repo} (is it public?)")
        z = zipfile.ZipFile(io.BytesIO(r.content))  # read in memory; never extracted to disk
        infos = {i.filename.split("/", 1)[1]: i for i in z.infolist() if not i.is_dir() and "/" in i.filename}
        get = lambda k: z.read(infos[k])  # noqa: E731
        sizes = {k: i.file_size for k, i in infos.items()}
        name = f"{owner}/{repo}"
    else:
        raise ValueError("Give a GitHub repo as owner/repo or https://github.com/owner/repo")

    keys = sorted(sizes)
    files, docs = {}, {}
    has_results = any(k.split("/")[0].lower() in RESULT_DIRS for k in keys)
    for k in keys:
        low, top, base = k.lower(), k.split("/")[0].lower(), k.rsplit("/", 1)[-1].lower()
        if sizes[k] > MAX_FILE:
            continue
        text = lambda: get(k).decode("utf-8", "replace")  # noqa: E731
        if "/" not in k and base.startswith("readme"):
            files[k] = text()
        elif low.endswith((".json", ".csv", ".md", ".txt")) and base not in SKIP_JSON and "live" not in base:
            if top in RESULT_DIRS or (not has_results and low.endswith(".json") and sizes[k] < 50_000):
                files[k] = text()
        elif low.endswith(".py") and not top.startswith(("test", ".")) and len(docs) < 30:
            mdoc = re.match(r'\s*(?:#.*\n|\s)*(?:"""|\'\'\')(.*?)(?:"""|\'\'\')', text(), re.DOTALL)
            if mdoc and mdoc.group(1).strip():
                docs[k] = mdoc.group(1).strip()[:1500]
    return Repo(name=name, files=files, docstrings=docs, tree=[k for k in keys if not k.startswith(".")][:300])


# ---------------------------------------------------------------- 2. facts
@dataclass
class Fact:
    id: int
    text: str
    source: str

    @property
    def numbers(self) -> list[float]:
        return numbers_in(self.text)


def numbers_in(text: str) -> list[float]:
    out = []
    for raw in NUM.findall(text):
        pct = raw.rstrip().endswith("%")
        try:
            v = float(raw.replace(",", "").replace("%", "").strip())
        except ValueError:
            continue
        out.append(v / 100 if pct else v)
        if pct:
            out.append(v)
    return out


def _flatten(x, path: str, out: list, depth: int = 0) -> None:
    if isinstance(x, dict):
        for k, v in x.items():
            _flatten(v, f"{path}.{k}" if path else str(k), out, depth + 1)
    elif isinstance(x, list):
        if x and all(not isinstance(v, (dict, list)) for v in x):
            out.append((depth, f"{path} = {', '.join(map(str, x[:15]))}{' …' if len(x) > 15 else ''}"))
        elif len(x) <= 6:
            for i, v in enumerate(x):
                _flatten(v, f"{path}[{i}]", out, depth + 1)
        else:
            out.append((depth, f"{path} = list of {len(x)} records"))
    elif x is not None:
        out.append((depth, f"{path} = {x}"))


def extract_facts(repo: Repo, per_file: int = 70) -> list[Fact]:
    facts: list[Fact] = []

    def add(text: str, src: str) -> None:
        text = re.sub(r"\s+", " ", text).strip()
        if text and not any(f.text == text for f in facts):
            facts.append(Fact(len(facts) + 1, text, src))

    for path, doc in repo.docstrings.items():  # what the code says it does: citable for Data / Methods
        for para in re.split(r"\n\s*\n", doc):
            if len(para.split()) >= 6:
                add(f"code ({path}): {para}", path)
    for path, text in repo.files.items():
        if path.lower().endswith(".json"):
            try:
                leaves: list = []
                _flatten(json.loads(text), "", leaves)
            except json.JSONDecodeError:
                continue
            for _, t in sorted(leaves, key=lambda t: t[0])[:per_file]:  # shallow (headline) numbers first
                add(f"{Path(path).stem}: {t}", path)
        elif path.lower().endswith(".csv"):
            rows = [r for r in text.splitlines() if r.strip()]
            if 1 < len(rows) <= 31:
                head = rows[0].split(",")
                for r in rows[1:]:
                    add(f"{Path(path).stem}: " + ", ".join(f"{h}={v}" for h, v in zip(head, r.split(","))), path)
        else:  # README / markdown: every line or table row that carries a number
            in_code, header, heading = False, None, ""
            lines = text.splitlines()
            cells_of = lambda s: [re.sub(r"[*_`]", "", c).strip() for c in s.strip().strip("|").split("|")]  # noqa: E731
            is_sep = lambda s: s.strip().startswith("|") and set(s.strip()) <= set("|-: ")  # noqa: E731
            for i, line in enumerate(lines):
                if line.strip().startswith("```"):
                    in_code = not in_code
                    continue
                if in_code:
                    continue
                if line.lstrip().startswith("#"):
                    heading = re.sub(r"[#*_`]", "", line).strip()
                if not line.strip().startswith("|"):
                    header = None
                if is_sep(line):
                    continue
                if line.strip().startswith("|") and i + 1 < len(lines) and is_sep(lines[i + 1]):
                    header = cells_of(line)  # the row above a |---| line is the table header
                    continue
                if not re.search(r"\d", line):
                    continue
                if line.strip().startswith("|"):
                    cells = cells_of(line)
                    row = " | ".join(f"{h}: {c}" for h, c in zip(header, cells)) if header and len(header) == len(cells) else " | ".join(cells)
                    add(f"{heading} — {row}" if heading else row, path)
                elif not re.match(r"\s*!\[|\s*<", line) and "http" not in line[:12]:
                    add(re.sub(r"[*`#>]", "", line), path)
    return facts


def number_ok(value: float, pool: list[float]) -> bool:
    """A number in the text matches a fact if equal at the text's precision, or as a percent of a fraction."""
    for p in pool:
        for cand in (p, p * 100, p / 100):
            tol = 0.5 * 10 ** -_decimals(value) + 1e-9
            if abs(cand - value) <= max(tol, abs(value) * 0.005):
                return True
    return False


def _decimals(v: float) -> int:
    s = repr(v)
    return len(s.split(".")[1].rstrip("0")) if "." in s and not s.endswith(".0") else 0


# ---------------------------------------------------------------- 3-6. prompts
PROFILE_SYS = """You read a software project's README and module descriptions and describe the research behind it.
Return JSON {"title":str,"one_line":str,"problem":str,"setting":str,"data":[str],"methods":[str],"baselines":[str],
"evaluation":str,"contributions":[str],"limitations":[str],"research_question":str,"keywords":[str]}.
- title: a paper title, specific and modest (no "novel", no hype).
- research_question: the research area as a literature-search question (about the field, not this repo).
- contributions: 3-4 concrete things this project does that a paper would claim, each one sentence.
- Use only what the text says. Do not invent results."""

SECTION_SYS = """You write ONE section of a research paper about the authors' own project. Rules:
- Facts about the project are numbered [F1], [F2]... Every sentence that states a number about the project
  must cite the fact(s) containing that exact number, e.g. "CSI reached 0.82 [F12]". Copy numbers exactly;
  do not compute new numbers (no new differences, ratios or averages).
- Published work is numbered [C1], [C2]... Claims about other work must cite the cards that state them.
- Describe a published paper only with what its line says; do not add models, regions, sensors or numbers
  that are not on that line. Cite each card separately, like [C2][C5]; cite facts like [F3][F4].
- Implementation details (hyperparameters, loss, augmentation, tuning, how data were split) must come from
  PROJECT or FACTS (facts starting "code (...)" describe what the code does; cite them). If a detail a reader
  needs is not given, write [TODO: what the author should add] instead of guessing.
- Cite only numbers that appear in the lists given. Never invent results, datasets, or references.
- Academic register, first-person plural ("we"), concise, no hype words (novel, groundbreaking, cutting-edge).
- Do not use markdown headings, bullets or bold. Plain paragraphs only. Write the full length asked for.
Return JSON with ONE key holding an array of paragraph strings: {"paragraphs":["first paragraph", "second paragraph"]}."""

SECTIONS = [
    ("Introduction", "motivation and why the problem matters (cite cards for every claim about prior work), what is missing in "
     "existing work, what this paper does, and a final paragraph listing the contributions in prose. 4 paragraphs, 350-450 words.", True, 10),
    ("Related Work", "group existing work by approach, say what each does and how well, compare them, and end with "
     "how this project differs. 3-4 paragraphs, 400-500 words. Cite cards for every claim about prior work; use at least 10 different cards.", True, 0),
    ("Data and Study Area", "region, period, data sources, labels/ground truth and preprocessing. 2 paragraphs.", False, 25),
    ("Methods", "the models, features, training procedure and the baselines, in enough detail to reproduce. "
     "3 paragraphs, 300-400 words.", False, 25),
    ("Evaluation Design", "how results were validated (splits, leakage control), metrics and their definitions, "
     "and baselines. 2 paragraphs, 200-300 words.", False, 20),
    ("Results", "the main quantitative results against baselines, at each setting reported (every lead time, every "
     "validation, every model). Every number must cite its fact. 4-5 paragraphs, 450-600 words.", False, 45),
    ("Discussion and Limitations", "what the results mean, where the method fails or is uncertain, threats to "
     "validity, and how results relate to published work (cite cards). 3 paragraphs, 300-400 words.", True, 20),
    ("Conclusion", "summary of what was shown and future work. 1-2 paragraphs, 150-200 words.", False, 12),
]

METHOD_SECTIONS = {"Data and Study Area", "Methods", "Evaluation Design"}

ABSTRACT_SYS = """Write a research paper abstract (150-220 words, one paragraph) from the verified sections given.
State the problem, approach, data, main quantitative results and what they mean. Keep the [F#] citations on every
number exactly as they appear in the sections; do not introduce numbers not present there. Return JSON {"abstract":str}."""

REVIEW_SYS = """You are a strict peer reviewer for an applied machine-learning venue. From the paper profile, the list
of available evidence and the draft, list what a reviewer would ask for before acceptance.
Return JSON {"strengths":[str],"missing":[{"item":str,"why":str,"how":str}]}. 3-5 strengths, 5-8 missing items,
concrete and specific to this project (e.g. confidence intervals, a stronger baseline, an ablation, data release)."""


@dataclass
class Paper:
    repo: str
    profile: dict
    facts: list[Fact]
    cards: list[Card]
    abstract: str
    sections: list[dict]
    flagged: list[dict]
    removed: list[dict]
    review: dict
    closest: list[dict]
    stats: dict

    def to_dict(self) -> dict:
        return {"repo": self.repo, "profile": self.profile, "facts": [asdict(f) for f in self.facts],
                "cards": [{"n": c.n, "score": c.score, "facts": c.facts, **c.paper.to_dict()} for c in self.cards],
                "abstract": self.abstract, "sections": self.sections, "flagged": self.flagged,
                "removed": self.removed, "review": self.review, "closest": self.closest, "stats": self.stats}


def cards_from_review(path: str | Path) -> list[Card]:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    names = set(S.Paper.__dataclass_fields__)
    return [Card(n=c["n"], score=c.get("score", 0), facts=c.get("facts", {}),
                 paper=S.Paper(**{k: v for k, v in c.items() if k in names})) for c in d["cards"]]


class PaperWriter:
    def __init__(self, llm, log: Callable[[str], None] = print, search_fns: dict | None = None):
        self.llm, self.log = llm, log
        self.scout = LitScout(llm, search_fns=search_fns, log=log, max_cards=20)

    # ---- profile
    def profile(self, repo: Repo) -> dict:
        readme = next((t for p, t in repo.files.items() if p.lower().startswith("readme")), "")
        docs = "\n".join(f"{p}: {d[:300]}" for p, d in list(repo.docstrings.items())[:15])
        try:
            prof = parse_json(self.llm.complete(PROFILE_SYS, f"README:\n{readme[:9000]}\n\nMODULES:\n{docs}", task="plan", json_mode=True, big=True))
        except (ValueError, json.JSONDecodeError):
            prof = {}
        prof.setdefault("title", repo.name)
        prof.setdefault("research_question", prof.get("one_line") or repo.name)
        for k in ("data", "methods", "baselines", "contributions", "limitations", "keywords"):
            prof[k] = [str(x) for x in prof.get(k, []) if x] if isinstance(prof.get(k), list) else []
        return prof

    # ---- pick facts relevant to a section (keeps each prompt inside free-tier limits)
    def pick_facts(self, facts: list[Fact], query: str, k: int, code: bool = False) -> list[Fact]:
        if k <= 0:
            return []
        q = tokens(query)
        boost = lambda f: (1.5 if f.source.lower().startswith("readme") else 0) + (6 if code and f.source.endswith(".py") else 0)  # noqa: E731
        scored = sorted(facts, key=lambda f: (-(len(q & tokens(f.text)) + boost(f)), f.id))
        return sorted(scored[:k], key=lambda f: f.id)

    def draft_section(self, name: str, brief: str, prof: dict, facts: list[Fact], cards: list[Card]) -> list[str]:
        fl = "\n".join(f"[F{f.id}] {f.text[:220]}" for f in facts) or "(none)"
        cl = "\n".join(f"[C{c.n}] {c.paper.title} ({c.paper.year}). {c.facts.get('method', '')}; "
                       f"finding: {c.facts.get('key_result', '')}"[:330] for c in cards) or "(none)"
        p = {k: prof.get(k) for k in ("title", "problem", "setting", "data", "methods", "baselines", "evaluation", "contributions", "limitations")}
        user = f"SECTION: {name}\nWHAT IT COVERS: {brief}\n\nPROJECT:\n{json.dumps(p, ensure_ascii=False)[:2500]}\n\nPROJECT FACTS:\n{fl}\n\nPUBLISHED WORK:\n{cl}"
        try:
            paras = paragraphs_from(self.llm.complete(SECTION_SYS, user, task="synthesize", json_mode=True, big=True))
        except (ValueError, json.JSONDecodeError, requests.RequestException):
            paras = []
        return paras

    # ---- check one sentence
    def check(self, sent: str, facts_by: dict[int, Fact], cards_by: dict[int, Card]) -> dict:
        fids, cids = [int(x) for x in FACT_CITE.findall(sent)], [int(x) for x in CARD_CITE.findall(sent)]
        bad = [f"F{i}" for i in fids if i not in facts_by] + [f"C{i}" for i in cids if i not in cards_by]
        if bad:
            return {"verdict": "invalid_citation", "reason": f"cites {', '.join(bad)}, which do not exist"}
        body = re.sub(r"[\u2010-\u2015\u2212]", "-", ANY_CITE.sub("", sent))
        body = re.sub(r"\(\d{1,2}\)|\b[A-Za-z]+-?\d+[A-Za-z0-9]*\b|\b\d[Dd]\b", " ", body)  # (1), ResNet-18, Sentinel-1, 2D
        nums = [n for n in numbers_in(body) if not (1900 <= n <= 2100 and float(n).is_integer())]
        if fids:
            pool = [x for i in fids for x in facts_by[i].numbers]
            missing = [n for n in nums if not number_ok(n, pool)] if not cids else \
                      [n for n in nums if not number_ok(n, pool) and not number_ok(n, numbers_in(" ".join(cards_by[i].paper.abstract for i in cids)))]
            if missing:
                return {"verdict": "number_mismatch", "reason": f"{', '.join(_fmt(n) for n in missing[:4])} not found in the cited evidence" + self._where(missing, facts_by)}
        if cids:
            v = self.scout.verify_sentence(re.sub(r"\[C(\d+)\]", r"[\1]", FACT_CITE.sub("", sent)), cards_by)
            return v if not fids or v["verdict"] != "unsupported" else {**v, "verdict": "partial"}
        if fids:
            return {"verdict": "matches_evidence"}
        if "[TODO" in sent:
            return {"verdict": "todo", "reason": "the author needs to add this detail"}
        if [n for n in nums if n > 10 or not float(n).is_integer()]:
            return {"verdict": "uncited_number", "reason": "states a number without citing evidence" + self._where([n for n in nums if n > 10 or not float(n).is_integer()], facts_by)}
        return {"verdict": "uncited"}

    @staticmethod
    def _where(nums: list[float], facts_by: dict[int, Fact]) -> str:
        """Point the author to the facts that do contain the numbers (README lines first)."""
        hits = []
        for n in nums[:3]:
            ids = sorted((i for i, f in facts_by.items() if number_ok(n, f.numbers)), key=lambda i: (not facts_by[i].source.lower().startswith("readme"), i))
            if ids:
                hits.append(f"{_fmt(n)} is in {', '.join(f'F{i}' for i in ids[:2])}")
        return f" ({'; '.join(hits)})" if hits else " (not found anywhere in the repo)"

    def check_paragraph(self, para: str, facts_by, cards_by, section: str, flagged: list, removed: list, tally: dict) -> str:
        para = normalise_cites(para)
        kept = []
        for s in [x.strip() for x in SENT.split(para) if x.strip()]:
            v = self.check(s, facts_by, cards_by)
            tally[v["verdict"]] = tally.get(v["verdict"], 0) + 1
            if v["verdict"] in ("invalid_citation", "unsupported"):
                removed.append({"section": section, "sentence": s, **v})
                continue
            if v["verdict"] == "uncited" and section in METHOD_SECTIONS and len(tokens(s)) >= 10:
                v = {"verdict": "uncited_claim", "reason": "a method detail with no evidence cited: check it is true"}
                tally["uncited"] -= 1
                tally["uncited_claim"] = tally.get("uncited_claim", 0) + 1
            if v["verdict"] in ("number_mismatch", "uncited_number", "partial", "uncited_claim", "todo"):
                flagged.append({"section": section, "sentence": s, **v})
            kept.append(s)
        return " ".join(kept)

    # ---- closest existing work per contribution (lexical, no model call)
    def closest(self, prof: dict, cards: list[Card]) -> list[dict]:
        out = []
        for c in prof.get("contributions", []):
            q = tokens(c)
            ranked = sorted(cards, key=lambda k: -len(q & tokens(k.paper.title + " " + k.paper.abstract)) / (len(q) or 1))
            out.append({"contribution": c, "closest": [{"n": k.n, "title": k.paper.title, "year": k.paper.year} for k in ranked[:2]]})
        return out

    def run(self, src: str, review_json: str | None = None) -> Paper:
        t0 = time.time()
        self.log(f"reading {src}")
        repo = read_repo(src)
        facts = extract_facts(repo)
        self.log(f"read {len(repo.files)} README/results files and {len(repo.docstrings)} code modules -> {len(facts)} evidence facts")
        if not facts:
            raise ValueError("No results found in the repo. Add a README with your numbers or a reports/ folder with metrics files.")
        prof = self.profile(repo)
        self.log(f"profile: {prof.get('title')}")
        if review_json:
            cards = cards_from_review(review_json)
            self.log(f"related work: reusing {len(cards)} papers from an earlier review")
        else:
            self.log(f"related work: searching the literature for '{prof['research_question'][:90]}'")
            cards = self.scout.run(prof["research_question"]).cards
        facts_by, cards_by = {f.id: f for f in facts}, {c.n: c for c in cards}
        sections, flagged, removed, tally = [], [], [], {}
        for name, brief, use_cards, k in SECTIONS:
            self.log(f"writing {name}")
            fs = self.pick_facts(facts, f"{name} {brief} {' '.join(prof.get('contributions', []))} {prof.get('evaluation', '')}", k,
                                 code=name in METHOD_SECTIONS)
            paras = self.draft_section(name, brief, prof, fs, cards if use_cards else [])
            self.log(f"checking {name}")
            paras = [self.check_paragraph(p, facts_by, cards_by, name, flagged, removed, tally) for p in paras]
            sections.append({"heading": name, "paragraphs": [p for p in paras if p]})
        self.log("writing the abstract")
        body = "\n\n".join(f"{s['heading']}: " + " ".join(s["paragraphs"]) for s in sections if s["heading"] in ("Introduction", "Results", "Conclusion"))
        try:
            abstract = str(parse_json(self.llm.complete(ABSTRACT_SYS, body[:7000], task="synthesize", json_mode=True, big=True)).get("abstract", ""))
        except (ValueError, json.JSONDecodeError, requests.RequestException):
            abstract = ""
        abstract = self.check_paragraph(abstract, facts_by, cards_by, "Abstract", flagged, removed, tally)
        self.log("reviewing the draft like a referee")
        try:
            evidence = "\n".join(f"{p}" for p in repo.files)[:1500]
            draft = (abstract + "\n" + "\n".join(" ".join(s["paragraphs"]) for s in sections))[:5000]
            review = parse_json(self.llm.complete(REVIEW_SYS, f"PROFILE: {json.dumps(prof, ensure_ascii=False)[:2000]}\n\nEVIDENCE FILES:\n{evidence}\n\nDRAFT:\n{draft}", task="plan", json_mode=True, big=True))
        except (ValueError, json.JSONDecodeError, requests.RequestException):
            review = {"strengths": [], "missing": []}
        used_f = sorted({int(x) for s in sections for p in s["paragraphs"] for x in FACT_CITE.findall(p)} | {int(x) for x in FACT_CITE.findall(abstract)})
        used_c = sorted({int(x) for s in sections for p in s["paragraphs"] for x in CARD_CITE.findall(p)})
        checked = sum(v for k, v in tally.items() if k != "uncited")
        stats = {"facts": len(facts), "facts_cited": len(used_f), "cards": len(cards), "cards_cited": len(used_c),
                 "sentences": sum(tally.values()), "verdicts": tally,
                 "numbers_matched_share": round(tally.get("matches_evidence", 0) / max(tally.get("matches_evidence", 0) + tally.get("number_mismatch", 0), 1), 3),
                 "checked_ok_share": round((tally.get("matches_evidence", 0) + tally.get("supported", 0)) / max(checked, 1), 3),
                 "seconds": round(time.time() - t0), "tokens": getattr(self.llm, "tokens", 0)}
        self.log("Done")
        return Paper(repo.name, prof, facts, cards, abstract, sections, flagged, removed,
                     {"strengths": [str(x) for x in review.get("strengths", [])], "missing": [m for m in review.get("missing", []) if isinstance(m, dict)]},
                     self.closest(prof, cards), stats)


GROUP = re.compile(r"\[([FC])\s*(\d+)((?:\s*[,;\u2010-\u2015-]\s*[FC]?\s*\d+)+)\]")


def normalise_cites(text: str) -> str:
    """[F16, F17] -> [F16][F17]; [F22-F26] -> [F22][F23]...[F26]; [C3, 5] -> [C3][C5]."""
    def expand(m):
        kind, ids, rest = m.group(1), [int(m.group(2))], m.group(3)
        for sep, k, n in re.findall(r"([,;\u2010-\u2015-])\s*([FC]?)\s*(\d+)", rest):
            n = int(n)
            if sep not in ",;" and n > ids[-1] and n - ids[-1] <= 12:
                ids += list(range(ids[-1] + 1, n + 1))
            else:
                ids.append(n)
        return "".join(f"[{kind}{i}]" for i in ids)
    return GROUP.sub(expand, text)


def paragraphs_from(text: str) -> list[str]:
    """Models sometimes repeat the "paragraphs" key once per paragraph; keep every one of them."""
    got: list[str] = []

    def hook(pairs):
        for k, v in pairs:
            if k in ("paragraphs", "paragraph", "text"):
                got.extend([v] if isinstance(v, str) else [x for x in v if isinstance(x, str)] if isinstance(v, list) else [])
        return dict(pairs)

    m = re.search(r"\{.*\}", text, re.DOTALL)
    json.loads(m.group(0) if m else text, object_pairs_hook=hook)
    return [x.strip() for x in got if x.strip()]


def _fmt(n: float) -> str:
    return str(int(n)) if float(n).is_integer() else f"{n:g}"


# ---------------------------------------------------------------- exports
def _bibkey(c: dict) -> str:
    last = (c.get("authors") or ["anon"])[0].split()[-1]
    return re.sub(r"[^A-Za-z0-9]", "", last).lower() + str(c.get("year") or "") + f"c{c['n']}"


def to_markdown(p: dict) -> str:
    lines = [f"# {p['profile'].get('title')}", "", "## Abstract", "", p["abstract"], ""]
    for s in p["sections"]:
        lines += [f"## {s['heading']}", ""] + [x + "\n" for x in s["paragraphs"]]
    used = sorted({int(x) for s in p["sections"] for q in s["paragraphs"] for x in CARD_CITE.findall(q)})
    by = {c["n"]: c for c in p["cards"]}
    lines += ["## References", ""] + [f"[C{n}] {', '.join(by[n]['authors'][:3])}{' et al.' if len(by[n]['authors']) > 3 else ''} ({by[n]['year']}). {by[n]['title']}. {by[n]['venue']}. {('https://doi.org/' + by[n]['doi']) if by[n]['doi'] else by[n]['url']}" for n in used if n in by]
    lines += ["", "## Evidence used", ""] + [f"[F{f['id']}] {f['text']} — `{f['source']}`" for f in p["facts"]
                                              if any(f"[F{f['id']}]" in q for s in p["sections"] for q in s["paragraphs"]) or f"[F{f['id']}]" in p["abstract"]]
    return "\n".join(lines)


UNI = {"\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2212": "-", "\u00a0": " ", "\u202f": " ", "\u2009": " ",
       "\u2248": "$\\approx$", "\u2265": "$\\geq$", "\u2264": "$\\leq$", "\u00d7": "$\\times$", "\u2192": "$\\rightarrow$",
       "\u00b1": "$\\pm$", "\u2032": "'", "\u2018": "`", "\u2019": "'", "\u201c": "``", "\u201d": "''"}


def _plain(s: str) -> str:
    return "".join(UNI.get(ch, ch) if not UNI.get(ch, "").startswith("$") else ch for ch in s)


def _tex(s: str) -> str:
    s = re.sub(r"([&%$#_{}])", r"\\\1", s)
    s = "".join(UNI.get(ch, ch) for ch in s)
    return s.replace("~", r"\textasciitilde{}").replace("^", r"\textasciicircum{}")


def to_latex(p: dict) -> tuple[str, str]:
    by = {c["n"]: c for c in p["cards"]}

    def conv(t: str) -> str:
        t = re.sub(r"\s*\[F\d+\]", "", t)          # own-evidence markers are for checking, not for print
        t = _tex(t)
        t = re.sub(r"\[C(\d+)\]((?:\s*\[C\d+\])*)", lambda m: r"~\cite{" + ",".join(_bibkey(by[int(n)]) for n in re.findall(r"C(\d+)", m.group(0)) if int(n) in by) + "}", t)
        return t

    body = [r"\documentclass[conference]{IEEEtran}", r"\usepackage{cite}", r"\begin{document}",
            r"\title{" + _tex(p["profile"].get("title", "")) + "}", r"\author{\IEEEauthorblockN{Author}}", r"\maketitle",
            r"\begin{abstract}" + conv(p["abstract"]) + r"\end{abstract}"]
    for s in p["sections"]:
        body += [r"\section{" + _tex(s["heading"]) + "}"] + [conv(x) + "\n" for x in s["paragraphs"]]
    body += [r"\bibliographystyle{IEEEtran}", r"\bibliography{refs}", r"\end{document}"]
    used = sorted({int(x) for s in p["sections"] for q in s["paragraphs"] for x in CARD_CITE.findall(q)})
    bib = []
    for n in used:
        c = by.get(n)
        if not c:
            continue
        fields = {"title": "{" + _plain(c["title"]).replace("&", "\\&") + "}", "author": _plain(" and ".join(c["authors"][:10])) or "Unknown",
                  "year": str(c.get("year") or ""), "journal": c.get("venue") or c.get("source", ""), "doi": c.get("doi", ""), "url": c.get("url", "")}
        bib.append(f"@article{{{_bibkey(c)},\n" + ",\n".join(f"  {k} = {{{v}}}" for k, v in fields.items() if v) + "\n}")
    return "\n".join(body), "\n\n".join(bib)


def main() -> None:
    ap = argparse.ArgumentParser(description="Draft a checked research paper from a project repo")
    ap.add_argument("repo", help="owner/repo, GitHub URL, or local folder")
    ap.add_argument("--review", help="reuse the papers from an earlier LitScout review.json for related work")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    p = PaperWriter(GroqLLM()).run(a.repo, a.review).to_dict()
    out = Path(a.out or ROOT / "reports" / "papers" / re.sub(r"[^a-z0-9]+", "-", p["repo"].lower()).strip("-"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "paper.json").write_text(json.dumps(p, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "paper.md").write_text(to_markdown(p), encoding="utf-8")
    tex, bib = to_latex(p)
    (out / "paper.tex").write_text(tex, encoding="utf-8")
    (out / "refs.bib").write_text(bib, encoding="utf-8")
    print(json.dumps(p["stats"], indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
