"""LLM clients. Groq (OpenAI-compatible) for real runs; a deterministic fake for tests.

Two model tiers: a small fast model for high-volume steps (screening, extraction) and a
larger one for planning, synthesis and verification, to stay inside free-tier limits.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Protocol

import requests


class LLM(Protocol):
    def complete(self, system: str, user: str, task: str, json_mode: bool = False, big: bool = False) -> str: ...


MAX_OUT = {"plan": 900, "screen": 700, "screen_alt": 700, "extract": 600, "synthesize": 2600, "verify": 350}
VERIFY_MODEL_ENV = "LITSCOUT_VERIFY_MODEL"
FALLBACKS = ["qwen/qwen3.8-27b", "openai/gpt-oss-120b", "openai/gpt-oss-20b"]
_EXHAUSTED: dict = {}  # (key, model) -> time when its quota frees up


class QuotaSpent(Exception):
    def __init__(self, model: str, wait: float):
        super().__init__(f"{model} quota spent, retry in {int(wait)}s")
        self.wait = wait
# Spread one review over three models: each has its own free daily token quota on Groq.
TASK_DEFAULTS = {"screen": "openai/gpt-oss-20b", "screen_alt": "qwen/qwen3.8-27b", "extract": "qwen/qwen3.8-27b",
                 "plan": "openai/gpt-oss-120b", "synthesize": "openai/gpt-oss-120b", "verify": "openai/gpt-oss-120b"}  # verification runs on a third model so its tokens/minute budget is separate


class GroqLLM:
    URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, small: str | None = None, big: str | None = None, api_key: str | None = None):
        self.small = small or os.getenv("LITSCOUT_SMALL_MODEL", "openai/gpt-oss-20b")
        self.big = big or os.getenv("LITSCOUT_BIG_MODEL", "openai/gpt-oss-120b")
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise RuntimeError("Set GROQ_API_KEY (free key from console.groq.com)")
        self.calls = 0
        self.tokens = 0
        self.cache = Path(os.getenv("LITSCOUT_LLM_CACHE", "data/llm_cache"))

    def complete(self, system: str, user: str, task: str, json_mode: bool = False, big: bool = False) -> str:
        model = self.big if big else self.small
        model = TASK_DEFAULTS.get(task, model)
        if task == "verify":
            model = os.getenv(VERIFY_MODEL_ENV, model)
        # per-task override, e.g. LITSCOUT_MODEL_EXTRACT=qwen/qwen3.8-27b (each model has its own daily quota)
        model = os.getenv(f"LITSCOUT_MODEL_{task.upper()}", model)
        h = hashlib.sha1(json.dumps([model, system, user, json_mode]).encode()).hexdigest()
        cf = self.cache / f"{h}.txt"
        if cf.exists():  # identical request seen before: reuse (temperature is 0)
            return cf.read_text(encoding="utf-8")
        self._task = task
        out = self._call(system, user, json_mode, model)
        self.cache.mkdir(parents=True, exist_ok=True)
        cf.write_text(out, encoding="utf-8")
        return out

    def _call(self, system: str, user: str, json_mode: bool, model: str) -> str:
        # Try the task's model first; if its free quota is spent (daily limit, or a long wait), fall back to the
        # other free models instead of stalling the review.
        chain = [model] + [m for m in FALLBACKS if m != model]
        last = None
        for m in chain:
            if _EXHAUSTED.get((self.api_key, m), 0) > time.time():
                continue
            try:
                return self._call_one(system, user, json_mode, m)
            except QuotaSpent as e:
                _EXHAUSTED[(self.api_key, m)] = time.time() + e.wait
                last = e
        raise RuntimeError(f"Groq rate limit: all free models are busy ({last})")

    def _call_one(self, system: str, user: str, json_mode: bool, model: str) -> str:
        body = {"model": model, "temperature": 0.0,
                # Groq counts max_completion_tokens against the 8k tokens/minute free limit: keep it tight per task
                "max_completion_tokens": MAX_OUT.get(self._task, 1500),
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        if model.startswith("openai/gpt-oss"):
            body["reasoning_effort"] = "low"
        for attempt in range(6):
            r = requests.post(self.URL, json=body, timeout=90, headers={"Authorization": f"Bearer {self.api_key}"})
            if r.status_code == 429:  # free-tier rate limit
                wait = float(r.headers.get("retry-after", 2 ** attempt * 3))
                if "per day" in r.text or "TPD" in r.text or wait > 30 or attempt >= 4:
                    raise QuotaSpent(model, max(wait, 60))
                time.sleep(wait)
                continue
            if r.status_code == 400 and "response_format" in body:
                # the model produced invalid JSON under JSON mode: retry once in plain mode and parse leniently
                body.pop("response_format")
                body["messages"][0]["content"] += "\nRespond with a single JSON object only."
                continue
            if r.status_code >= 500 and attempt < 5:
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            d = r.json()
            self.calls += 1
            self.tokens += d.get("usage", {}).get("total_tokens", 0)
            return d["choices"][0]["message"]["content"]
        raise RuntimeError("Groq request failed after retries")


def parse_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            raise
        return json.loads(m.group(0))


class FakeLLM:
    """Deterministic stand-in: exercises every pipeline branch without a network."""

    def __init__(self, hallucinate: bool = False):
        self.calls: list[str] = []
        self.hallucinate = hallucinate

    def complete(self, system: str, user: str, task: str, json_mode: bool = False, big: bool = False) -> str:
        self.calls.append(task)
        if task == "plan":
            q = user.split("QUESTION:", 1)[-1].strip().splitlines()[0]
            return json.dumps({"subtopics": [{"name": "Core methods", "queries": [q]}], "inclusion": "on topic"})
        if task.startswith("screen"):
            ids = re.findall(r"^\[(\d+)\]", user, re.M)
            return json.dumps({"scores": [{"i": int(i), "score": 3, "reason": "relevant"} for i in ids]})
        if task == "extract":
            abstract = user.split("ABSTRACT:", 1)[-1].strip()
            first = re.split(r"(?<=\.)\s", abstract)[0] if abstract else ""
            return json.dumps({"task": "forecasting", "method": "model", "data": "data", "region": "",
                               "key_result": first, "evidence_quote": first})
        if task == "synthesize":
            n = re.findall(r"^\[(\d+)\]", user, re.M)
            body = " ".join(f"Study {k} reports its main finding [{k}]." for k in n[:3])
            if self.hallucinate:
                body += " Quantum blockchain flood oracles achieve perfect accuracy [99]."
            return json.dumps({"sections": [{"heading": "Core methods", "text": body}], "gaps": ["No study does X [1]."]})
        if task == "verify":
            return json.dumps({"verdict": "supported", "reason": "fake"})
        return "{}"
