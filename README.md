# LitScout

An AI research agent that writes literature reviews where **every citation is a real paper and every cited sentence is checked against its source**.

**Demo:** https://samiyakazi23.github.io/litscout/ · example reviews with clickable citations, the evidence behind each one, and the sentences the verifier cut.

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

Tests (`tests/`) include a model that deliberately invents a citation and a claim: both are always removed.

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

### Running on free tiers

Groq's free tier allows 8,000 tokens per minute and 200,000 per day *per model*, and counts the maximum answer length against the minute limit. LitScout:
- caps output length per step;
- routes steps to different models so they draw on separate quotas (`LITSCOUT_MODEL_<TASK>`, defaults: gpt-oss-20b for screening, Qwen for extraction, gpt-oss-120b for planning, writing and verification);
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
docs/           demo site (GitHub Pages)
tests/          offline tests, including invented-citation and paraphrased-quote checks
```

## Limitations

- Verification uses abstracts, not full texts, so claims about details only in a paper's body are marked partial or cut.
- Recall against survey reference lists is a strict measure: surveys also cite background, datasets and methods outside the topic.
- An LLM judge can be wrong in both directions; the removed-sentence list is shown so readers can check.

Built by [Samiya Kazi](https://github.com/samiyakazi23). MIT licence.
