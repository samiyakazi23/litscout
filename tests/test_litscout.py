import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from litscout import sources as S  # noqa: E402
from litscout.agent import LitScout, to_markdown, tokens  # noqa: E402
from litscout.llm import FakeLLM, parse_json  # noqa: E402

CORPUS = [
    S.Paper("openalex:W1", "LSTM flood forecasting on the Brahmaputra", "We forecast river discharge with LSTM networks. Lead times of 1-5 days reach NSE 0.9.", 2021, ["A"], "J Hydrol", "10.1/a", citations=50),
    S.Paper("openalex:W2", "Sentinel-1 flood mapping with U-Net", "A U-Net maps floodwater from SAR imagery. It reaches IoU 0.6 on Sen1Floods11.", 2020, ["B"], "RSE", "10.1/b", citations=80),
    S.Paper("arxiv:2101.1", "LSTM flood forecasting on the Brahmaputra", "Duplicate preprint of W1 with a shorter abstract.", 2021, ["A"], "arXiv", "10.1/a"),
    S.Paper("openalex:W3", "Graph networks for streamflow", "Graph neural networks route flow between gauges. They outperform LSTMs at 7 days.", 2023, ["C"], "WRR", "", citations=10),
    S.Paper("openalex:W4", "Deep learning for crop yield", "Crop yield prediction from weather.", 2022, ["D"], "Agri", ""),
]


def fake_search(q, n=20):
    return [S.Paper(**{**p.to_dict(), "found_by": [q]}) for p in CORPUS]


def agent(llm=None, **kw):
    a = LitScout(llm or FakeLLM(), search_fns={"fake": fake_search}, log=lambda *_: None, **kw)
    a.snowball = lambda kept: []  # no network in tests
    return a


def test_merge_dedupes_by_doi_and_keeps_richer_record():
    m = S.merge(CORPUS)
    assert len(m) == 4
    w1 = next(p for p in m if p.doi == "10.1/a")
    assert "NSE 0.9" in w1.abstract and w1.citations == 50


def test_openalex_abstract_rebuilt_in_order():
    assert S._oa_abstract({"floods": [1], "Assam": [0], "worsen": [2]}) == "Assam floods worsen"


def test_full_run_produces_verified_review():
    r = agent().run("flood forecasting with deep learning")
    assert r.cards and r.sections
    assert r.stats["invalid_citations_removed"] == 0
    md = to_markdown(r)
    assert "## References" in md and "## Comparison" in md


def test_invented_citation_never_survives():
    r = agent(FakeLLM(hallucinate=True)).run("flood forecasting with deep learning")
    text = " ".join(s["text"] for s in r.sections)
    assert "[99]" not in text and "Quantum" not in text
    assert any(x.get("verdict") == "invalid_citation" for x in r.removed)


def test_evidence_quote_must_be_verbatim():
    a = agent()

    class Paraphraser(FakeLLM):
        def complete(self, system, user, task, json_mode=False, big=False):
            if task == "extract":
                return '{"key_result": "LSTMs are great", "evidence_quote": "LSTMs are really great at everything"}'
            return super().complete(system, user, task, json_mode, big)

    a.llm = Paraphraser()
    assert a.extract(CORPUS[0])["evidence_quote"] == ""


def test_year_window_respected():
    r = agent(until_year=2021).run("flood forecasting")
    assert all(c.paper.year <= 2021 for c in r.cards)


def test_unsupported_sentence_removed():
    class Judge(FakeLLM):
        def complete(self, system, user, task, json_mode=False, big=False):
            if task == "verify":
                return '{"verdict": "unsupported", "reason": "not in source"}'
            return super().complete(system, user, task, json_mode, big)

    r = agent(Judge()).run("flood forecasting")
    assert not r.sections and r.removed


def test_helpers():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert "flood" in tokens("The flood forecasting study")
