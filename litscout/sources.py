"""Paper search across arXiv, OpenAlex and Semantic Scholar, with citation snowballing.

Every paper carries a stable id from its source (arXiv id, OpenAlex W-id, S2 id, DOI),
so a citation in the final review can always be resolved back to a real record.
Responses are cached on disk: reruns are fast and polite to the APIs.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path

import requests

import os

CACHE = Path("data/cache")
OPENALEX_KEY = os.getenv("OPENALEX_API_KEY", "")
UA = {"User-Agent": "LitScout/0.1 (research literature agent)"}


@dataclass
class Paper:
    pid: str                      # source-prefixed id, e.g. "arxiv:2310.07631", "openalex:W123", "doi:10.1/x"
    title: str
    abstract: str = ""
    year: int | None = None
    authors: list[str] = field(default_factory=list)
    venue: str = ""
    doi: str = ""
    url: str = ""
    citations: int = 0
    source: str = ""
    found_by: list[str] = field(default_factory=list)  # queries / snowball steps that surfaced it
    refs: list[str] = field(default_factory=list)       # OpenAlex ids of referenced works, when known

    def key(self) -> str:
        """Dedup key: DOI if present, else normalised title."""
        return ("doi:" + self.doi.lower()) if self.doi else "t:" + re.sub(r"[^a-z0-9]", "", self.title.lower())[:120]

    def to_dict(self) -> dict:
        return asdict(self)


def _cached_get(url: str, params: dict | None = None, kind: str = "json", retries: int = 4, sleep: float = 0.0):
    CACHE.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha1((url + json.dumps(params or {}, sort_keys=True)).encode()).hexdigest()  # key excluded from cache id
    headers = dict(UA)
    if OPENALEX_KEY and "openalex.org" in url:
        headers["Authorization"] = f"Bearer {OPENALEX_KEY}"
    f = CACHE / f"{h}.{kind}"
    if f.exists():
        txt = f.read_text(encoding="utf-8")
        return json.loads(txt) if kind == "json" else txt
    last = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=45)
            if r.status_code == 429:
                time.sleep(3 * 2 ** attempt)
                continue
            r.raise_for_status()
            f.write_text(r.text, encoding="utf-8")
            if sleep:
                time.sleep(sleep)
            return r.json() if kind == "json" else r.text
        except requests.RequestException as e:
            last = e
            time.sleep(2 * 2 ** attempt)
    raise RuntimeError(f"request failed: {url} ({last})")


# ---------------- arXiv ----------------
ARXIV = "https://export.arxiv.org/api/query"
NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


def search_arxiv(query: str, n: int = 20) -> list[Paper]:
    terms = " AND ".join(f'all:"{t}"' if " " in t else f"all:{t}" for t in _split_query(query))
    xml = _cached_get(ARXIV, {"search_query": terms, "max_results": n, "sortBy": "relevance"}, kind="xml", sleep=3)
    root = ET.fromstring(xml)
    out = []
    for e in root.findall("a:entry", NS):
        aid = e.findtext("a:id", "", NS).rsplit("/abs/", 1)[-1]
        aid = re.sub(r"v\d+$", "", aid)
        doi = e.findtext("arxiv:doi", "", NS) or ""
        out.append(Paper(
            pid=f"arxiv:{aid}", title=" ".join(e.findtext("a:title", "", NS).split()),
            abstract=" ".join(e.findtext("a:summary", "", NS).split()),
            year=int(e.findtext("a:published", "0000", NS)[:4]),
            authors=[a.findtext("a:name", "", NS) for a in e.findall("a:author", NS)],
            venue="arXiv", doi=doi.lower(), url=f"https://arxiv.org/abs/{aid}", source="arxiv", found_by=[query]))
    return out


def _split_query(q: str) -> list[str]:
    """Keep quoted phrases, split the rest into words (arXiv search is term-based)."""
    phrases = re.findall(r'"([^"]+)"', q)
    rest = re.sub(r'"[^"]+"', " ", q)
    words = [w for w in re.findall(r"[A-Za-z0-9\-]+", rest) if len(w) > 2 and w.lower() not in {"and", "the", "for", "with", "using", "from"}]
    return phrases + words


# ---------------- OpenAlex ----------------
OPENALEX = "https://api.openalex.org/works"


def _oa_abstract(inv: dict | None) -> str:
    if not inv:
        return ""
    pos = {}
    for word, idxs in inv.items():
        for i in idxs:
            pos[i] = word
    return " ".join(pos[i] for i in sorted(pos))


def _oa_paper(w: dict, found_by: str) -> Paper:
    doi = (w.get("doi") or "").replace("https://doi.org/", "").lower()
    src = ((w.get("primary_location") or {}).get("source") or {})
    return Paper(
        pid="openalex:" + w["id"].rsplit("/", 1)[-1], title=w.get("display_name") or w.get("title") or "",
        abstract=_oa_abstract(w.get("abstract_inverted_index")), year=w.get("publication_year"),
        authors=[a["author"]["display_name"] for a in (w.get("authorships") or [])[:12] if a.get("author")],
        venue=src.get("display_name") or "", doi=doi, url=("https://doi.org/" + doi) if doi else w["id"],
        citations=w.get("cited_by_count") or 0, source="openalex", found_by=[found_by],
        refs=[r.rsplit("/", 1)[-1] for r in (w.get("referenced_works") or [])])


def search_openalex(query: str, n: int = 25, from_year: int | None = None) -> list[Paper]:
    params = {"search": query, "per-page": min(n, 50), "mailto": "litscout@users.noreply.github.com"}
    if from_year:
        params["filter"] = f"from_publication_date:{from_year}-01-01"
    data = _cached_get(OPENALEX, params)
    return [_oa_paper(w, query) for w in data.get("results", []) if w.get("display_name")]


def openalex_works(ids: list[str], found_by: str) -> list[Paper]:
    """Fetch works by OpenAlex id, 50 per request."""
    out = []
    for i in range(0, len(ids), 50):
        chunk = ids[i:i + 50]
        data = _cached_get(OPENALEX, {"filter": "openalex:" + "|".join(chunk), "per-page": 50, "mailto": "litscout@users.noreply.github.com"})
        out += [_oa_paper(w, found_by) for w in data.get("results", [])]
    return out


def openalex_citing(work_id: str, n: int = 25) -> list[Paper]:
    data = _cached_get(OPENALEX, {"filter": f"cites:{work_id}", "per-page": n, "sort": "cited_by_count:desc", "mailto": "litscout@users.noreply.github.com"})
    return [_oa_paper(w, f"cites:{work_id}") for w in data.get("results", [])]


def openalex_resolve(doi: str = "", title: str = "") -> Paper | None:
    """Look a paper up by DOI or exact-ish title: used to verify that a citation is real."""
    if doi:
        try:
            w = _cached_get(f"{OPENALEX}/https://doi.org/{doi}", {"mailto": "litscout@users.noreply.github.com"})
            return _oa_paper(w, "resolve")
        except RuntimeError:
            pass
    if title:
        data = _cached_get(OPENALEX, {"filter": f"title.search:{re.sub(r'[,:|]', ' ', title)[:200]}", "per-page": 3, "mailto": "litscout@users.noreply.github.com"})
        for w in data.get("results", []):
            if _norm(w.get("display_name") or "") == _norm(title):
                return _oa_paper(w, "resolve")
    return None


def _norm(t: str) -> str:
    return re.sub(r"[^a-z0-9]", "", t.lower())


# ---------------- Semantic Scholar (optional; rate-limited without a key) ----------------
S2 = "https://api.semanticscholar.org/graph/v1/paper/search"


def search_s2(query: str, n: int = 20) -> list[Paper]:
    fields = "title,abstract,year,authors,venue,externalIds,citationCount,url"
    try:
        data = _cached_get(S2, {"query": query, "limit": n, "fields": fields}, retries=2)
    except RuntimeError:
        return []
    out = []
    for p in data.get("data", []):
        ext = p.get("externalIds") or {}
        doi = (ext.get("DOI") or "").lower()
        out.append(Paper(pid=f"s2:{p['paperId']}", title=p.get("title") or "", abstract=p.get("abstract") or "", year=p.get("year"),
                         authors=[a["name"] for a in (p.get("authors") or [])[:12]], venue=p.get("venue") or "", doi=doi,
                         url=p.get("url") or "", citations=p.get("citationCount") or 0, source="s2", found_by=[query]))
    return out


def merge(papers: list[Paper]) -> list[Paper]:
    """Deduplicate across sources, keeping the richest record and the union of provenance."""
    best: dict[str, Paper] = {}
    for p in papers:
        k = p.key()
        if k not in best:
            best[k] = p
            continue
        q = best[k]
        q.found_by = sorted(set(q.found_by) | set(p.found_by))
        if len(p.abstract) > len(q.abstract):
            q.abstract = p.abstract
        q.citations = max(q.citations, p.citations)
        q.refs = q.refs or p.refs
        if not q.doi and p.doi:
            q.doi = p.doi
        if q.source == "arxiv" and p.source == "openalex":  # prefer the published version's metadata
            q.venue, q.pid = p.venue or q.venue, p.pid
    return list(best.values())
