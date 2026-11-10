# LitScout

An AI research agent that writes literature reviews where **every citation is a real paper and every cited sentence is checked against its source**.

**Try it:** https://huggingface.co/spaces/samiyakazi23/litscout · type any research question and get a verified review in about 10 minutes, or paste a GitHub repo and get a checked draft paper.
**Static demo:** https://samiyakazi23.github.io/litscout/ · example reviews with clickable citations, the evidence behind each one, and the sentences the verifier cut.

AI tools that write literature reviews are known for inventing papers and misquoting real ones. LitScout is built so that it cannot cite anything it has not retrieved, and so that a claim its sources do not support never reaches the reader.

## What it does

```
question ─► plan (sub-topics, search queries)
         ─► search OpenAlex + arXiv (+ Semantic Scholar when available)
         ─► rank lexically, then LLM-screen for relevance (0–3 rubric)
         ─► snowball: references and citing papers of the strongest seeds, screened again
         ─► extract per paper: task, method, data, region, key result + a verbatim quote
                (a quote not found word-for-word in the abstract is discarded)
         ─► write from numbered evidence cards only
         ─► verify every sentence:
                cites a card that doesn't exist  → removed (invented citation)
                LLM judge vs cited abstracts     → supported / partial / unsupported
                lexical-overlap floor            → guards against an over-generous judge
         ─► review: sections, research gaps, comparison table, references
```

## Results

### Finding what experts cite

For each of 7 published survey papers (flood forecasting, SAR remote sensing, multimodal LLMs, financial forecasting, crop yield, landslides, graph neural networks), the gold set is the survey's reference list. LitScout gets only the survey's topic and its publication year as a cut-off. Every method is ranked by the same scorer and cut to the same size, so differences come from what the search *finds*.

| Method | Recall in top 50 | top 100 | top 200 |
|---|---|---|---|
| One keyword search (baseline) | 1.8% | 2.3% | 2.3% |
| LitScout planned queries | 3.5% | 4.6% | 5.6% |
| **LitScout + citation snowballing** | **6.3%** | **9.5%** | **14.6%** |

Per survey (top 200): Evaluation of Machine Learning approach in flood prediction … 2% → 6%; FUSING OF OPTICAL AND SYNTHETIC APERTURE RADAR (SAR) REMOTE … 0% → 16%; A survey on multimodal large language models 2% → 21%; Financial time series forecasting with deep learning : A sys… 0% → 15%; Crop yield prediction using machine learning: A systematic l… 1% → 12%; Application of Artificial Intelligence and Remote Sensing fo… 8% → 8%; A Comprehensive Survey on Graph Neural Networks 4% → 25%.

LitScout finds about 6× more of the experts' references than a single search (14.6% vs 2.3% in the top 200), and snowballing beats plain search in 6 of 7 surveys. Absolute recall is modest because survey reference lists include background, datasets and methods well outside the topic; the comparison between methods is what the benchmark measures.

### Verification on real reviews

| Review | Papers found | Kept | Cited sentences fully supported | Invented citations reaching the reader |
|---|---|---|---|---|
| ML for flood forecasting in the Brahmaputra basin | 731 | 25 | 7 of 12 (3 partly supported; 2 unsupported, cut) | 0 |
| ML flood forecasting in large basins (arXiv only) | 90 | 15 | 14 of 14 | 0 |
| Deep learning for landslide detection (run through the public service) | 635 | 25 | 17 of 27 (6 partly supported; 3 unsupported, cut) | 0 |

Tests (`tests/`) include a model that deliberately invents a citation and a claim: both are always removed.

## Paper mode: from your project to a draft paper

Give LitScout a public GitHub repo and it drafts a research paper about the project, then checks it.

```
repo ─► read README, results files (reports/, results/ …) and module docstrings; never runs the code
     ─► every number and every code description becomes a numbered fact [F#] tied to its file
     ─► profile: problem, data, method, contributions; related work from a LitScout review [C#]
     ─► draft Introduction, Related Work, Data, Methods, Evaluation, Results, Discussion, Conclusion, Abstract
     ─► check every sentence:
            a number about the project must appear in the fact it cites   → else flagged, with the fact that has it
            a number with no citation                                     → flagged
            a method detail with no evidence                              → flagged; unknown details become [TODO]
            a claim about other work                                      → judged against the cited abstracts
            a citation to a fact or paper that does not exist             → removed
     ─► referee checklist, closest existing work per contribution, Markdown / LaTeX (IEEE) / BibTeX
```

```bash
python -m litscout.paper samiyakazi23/floodwatch                      # searches related work itself
python -m litscout.paper samiyakazi23/floodwatch --review docs/reviews/brahmaputra-floods.json   # reuse a review
```

On [FloodWatch](https://github.com/samiyakazi23/floodwatch) the draft has 138 sentences: 80 numbers matched to the repo and 2 flagged as not matching their cited fact; 4 claims about published work fully supported and 11 partly (shown highlighted); 25 sentences flagged for the author in total. It used about 26k tokens with a reused review. The LaTeX export compiles to a 10-page draft.

A draft is a starting point for the author, not a finished paper: the flagged sentences and [TODO] markers are the parts only the author can confirm.

## Run it

```bash
pip install -r requirements.txt
export GROQ_API_KEY=...          # free at console.groq.com
export OPENALEX_API_KEY=...      # free at openalex.org
python -m litscout.cli "your research question" --until 2025
pytest -q                        # offline tests, no keys needed
python -m litscout.evaluate      # survey-recall benchmark (data/benchmark/surveys.json)
```

Outputs go to `reports/reviews/<question>/review.md` and `review.json`.

### Public web service

```bash
pip install -r requirements-server.txt
uvicorn litscout.server:app --port 7860     # open http://localhost:7860
```

Anyone can type a question; the page shows each step live (plan, search, screen, snowball, read, write, verify) and adds the finished review to a public gallery. The host's key is limited to `DAILY_CAP` reviews a day (default 4) and one review runs at a time. Visitors can paste their own free Groq key, which is used only for their run and never stored. The `Dockerfile` runs it on a Hugging Face Space (`scripts/deploy_hf.sh`), with `GROQ_API_KEY` and `OPENALEX_API_KEY` set as Space secrets.

### Running on free tiers

Groq's free tier allows 8,000 tokens per minute and 200,000 per day *per model*, and counts the maximum answer length against the minute limit. LitScout:
- caps output length per step;
- routes steps to different models so they draw on separate quotas (`LITSCOUT_MODEL_<TASK>`, defaults: gpt-oss-20b and Qwen alternating for screening, Qwen for extraction, gpt-oss-120b for planning, writing and verification);
- when a model's daily quota runs out, switches to the next free model instead of waiting;
- queries the paper sources in parallel and stops asking a source that is rate-limiting;
- caches every model response and API call on disk, so reruns cost nothing.

A full review uses roughly 60,000 tokens and takes 5–10 minutes on the free tier.

## Layout

```
litscout/
  sources.py    arXiv, OpenAlex, Semantic Scholar clients; dedup; snowballing; citation resolution
  agent.py      plan → search → screen → snowball → extract → write → verify
  llm.py        Groq client with per-task routing, token caps, retries, cache; FakeLLM for tests
  evaluate.py   survey reference-list recall benchmark
  cli.py        command line
  paper.py      Paper mode: repo -> evidence facts -> checked draft paper -> Markdown / LaTeX / BibTeX
  server.py     public web service (FastAPI): reviews and papers, live progress, daily cap, bring-your-own-key
docs/           demo site (GitHub Pages)
tests/          offline tests, including invented-citation and paraphrased-quote checks
```

## Limitations

- Verification uses abstracts, not full texts, so claims about details only in a paper's body are marked partial or cut.
- Recall against survey reference lists is a strict measure: surveys also cite background, datasets and methods outside the topic.
- An LLM judge can be wrong in both directions; the removed-sentence list is shown so readers can check.

Built by [Samiya Kazi](https://github.com/samiyakazi23). MIT licence.
