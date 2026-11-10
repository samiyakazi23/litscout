"""Run a review with the sources available here (OpenAlex + arXiv; Semantic Scholar if reachable)."""
import json, sys, time, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from litscout.agent import LitScout, to_markdown
from litscout.llm import GroqLLM
from litscout import sources as S
from litscout.cli import slug

q = sys.argv[1]
llm = GroqLLM(); t = time.time()
fns = {"openalex": S.search_openalex, "arxiv": lambda x: S.search_arxiv(x, 12)}
if os.getenv("USE_S2") == "1":
    fns["s2"] = S.search_s2
agent = LitScout(llm, search_fns=fns, max_cards=int(os.getenv("MAX_CARDS", "25")))
r = agent.run(q)
r.stats.update(seconds=round(time.time() - t), tokens=llm.tokens)
out = Path("reports/reviews") / slug(q); out.mkdir(parents=True, exist_ok=True)
(out / "review.md").write_text(to_markdown(r), encoding="utf-8")
(out / "review.json").write_text(json.dumps(r.to_dict(), indent=1, ensure_ascii=False), encoding="utf-8")
print(json.dumps(r.stats, indent=1)); print("->", out)
