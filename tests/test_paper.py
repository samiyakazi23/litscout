import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from litscout import sources as S  # noqa: E402
from litscout.agent import Card  # noqa: E402
from litscout.llm import FakeLLM  # noqa: E402
from litscout.paper import PaperWriter, extract_facts, number_ok, read_repo, to_latex, to_markdown  # noqa: E402

README = """# Demo
Flood model tested on 2021-2026.

| Lead time | CSI | Baseline |
|---|---|---|
| 1 day | **0.82** | 0.73 |
"""


def make_repo(tmp_path):
    (tmp_path / "README.md").write_text(README)
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "metrics.json").write_text(json.dumps({"test": {"iou": 0.6512, "n": 120}}))
    (tmp_path / "pkg.py").write_text('"""Trains the flood model."""\nimport os\n')
    return tmp_path


def writer():
    class Judge(FakeLLM):
        def complete(self, system, user, task, json_mode=False, big=False):
            if task == "verify":
                return '{"verdict": "supported", "reason": ""}'
            return super().complete(system, user, task, json_mode, big)
    return PaperWriter(Judge(), log=lambda m: None)


def test_repo_facts_keep_table_headers_and_files(tmp_path):
    repo = read_repo(str(make_repo(tmp_path)))
    facts = extract_facts(repo)
    assert "pkg.py" in repo.docstrings
    assert any("CSI: 0.82" in f.text and "Baseline: 0.73" in f.text for f in facts)
    assert any(f.source == "reports/metrics.json" and "0.6512" in f.text for f in facts)


def test_number_matching():
    assert number_ok(0.65, [0.6512]) and number_ok(65, [0.6512]) and number_ok(0.82, [0.82])
    assert not number_ok(0.9, [0.82]) and not number_ok(14, [0.6512])


def test_checker_flags_wrong_numbers_and_cuts_fake_citations(tmp_path):
    w = writer()
    facts = extract_facts(read_repo(str(make_repo(tmp_path))))
    fb = {f.id: f for f in facts}
    csi = next(f.id for f in facts if "CSI: 0.82" in f.text)
    card = Card(1, S.Paper("openalex:W1", "LSTM floods", "LSTMs forecast floods with NSE 0.9.", 2021, ["A B"]), 3)
    cb = {1: card}
    assert w.check(f"Our model reached a CSI of 0.82 [F{csi}].", fb, cb)["verdict"] == "matches_evidence"
    assert w.check(f"Our model reached a CSI of 0.91 [F{csi}].", fb, cb)["verdict"] == "number_mismatch"
    assert w.check("Our model reached a CSI of 0.91.", fb, cb)["verdict"] == "uncited_number"
    assert w.check("LSTMs forecast floods [C7].", fb, cb)["verdict"] == "invalid_citation"
    assert w.check("A fact we made up [F999].", fb, cb)["verdict"] == "invalid_citation"
    flagged, removed, tally = [], [], {}
    text = w.check_paragraph(f"CSI was 0.82 [F{csi}]. Prior work did more [C9]. CSI was 0.5 [F{csi}].", fb, cb, "Results", flagged, removed, tally)
    assert "[C9]" not in text and len(removed) == 1 and len(flagged) == 1


def test_exports():
    p = {"profile": {"title": "T & co"}, "abstract": "We reach 0.82 [F1].",
         "sections": [{"heading": "Intro", "paragraphs": ["LSTMs work [C1]. Ours is 0.82 [F1]."]}],
         "cards": [{"n": 1, "authors": ["Ann Lee"], "year": 2021, "title": "LSTM floods", "venue": "J", "doi": "10.1/a", "url": "", "source": "openalex"}],
         "facts": [{"id": 1, "text": "CSI 0.82", "source": "README.md"}]}
    tex, bib = to_latex(p)
    assert r"\cite{lee2021c1}" in tex and "[F1]" not in tex and r"T \& co" in tex
    assert "@article{lee2021c1" in bib
    assert "[C1] Ann Lee (2021)" in to_markdown(p)


def test_citation_groups_are_split():
    from litscout.paper import normalise_cites
    assert normalise_cites("a [F2-F4] b [F7, F9] c [C1; C3]") == "a [F2][F3][F4] b [F7][F9] c [C1][C3]"
