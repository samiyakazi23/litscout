"""Public LitScout service: a search box anyone can use.

    uvicorn litscout.server:app --port 7860

- POST /api/review {question, groq_key?} starts a review in the background and returns a job id.
- GET  /api/job/{id} returns status, live progress lines, and the review when done.
- GET  /api/reviews lists finished reviews (the public gallery); GET /api/reviews/{slug} returns one.
- The host's key is capped at DAILY_CAP reviews per day; visitors can bring their own Groq key,
  which is used only for their run and never stored or logged.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from datetime import date
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .agent import LitScout
from .llm import GroqLLM
from .paper import PaperWriter, to_latex, to_markdown

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
STORE = Path(os.getenv("LITSCOUT_STORE", ROOT / "data" / "public"))
DAILY_CAP = int(os.getenv("DAILY_CAP", "4"))
MAX_ACTIVE = int(os.getenv("MAX_ACTIVE", "1"))  # free-tier limits are per minute: run one review at a time

app = FastAPI(title="LitScout")
jobs: dict[str, dict] = {}
lock = threading.Lock()
usage = {"day": date.today().isoformat(), "count": 0}


class ReviewRequest(BaseModel):
    question: str = Field(min_length=12, max_length=300)
    groq_key: str | None = Field(default=None, max_length=200)
    until_year: int | None = Field(default=None, ge=1950, le=2100)


class PaperRequest(BaseModel):
    repo: str = Field(min_length=3, max_length=200, pattern=r"^(https://github\.com/)?[\w.-]+/[\w.-]+/?$")
    groq_key: str | None = Field(default=None, max_length=200)


PAPER_COST = 2  # a paper runs a literature review plus drafting: counts as two free runs


def slug(q: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", q.lower()).strip("-")[:60] + "-" + uuid.uuid4().hex[:6]


def _host_quota_ok(cost: int = 1) -> bool:
    today = date.today().isoformat()
    if usage["day"] != today:
        usage.update(day=today, count=0)
    return usage["count"] + cost <= DAILY_CAP


def _run(job_id: str, req: ReviewRequest, key: str) -> None:
    job = jobs[job_id]

    def log(msg: str) -> None:
        job["progress"].append({"t": round(time.time() - job["started"]), "msg": msg})

    try:
        llm = GroqLLM(api_key=key)
        agent = LitScout(llm, until_year=req.until_year, log=log)
        log("Planning sub-topics and search queries")
        review = agent.run(req.question)
        review.stats.update(seconds=round(time.time() - job["started"]), tokens=llm.tokens)
        data = review.to_dict()
        data["slug"] = job["slug"]
        data["created"] = time.strftime("%Y-%m-%d")
        STORE.mkdir(parents=True, exist_ok=True)
        (STORE / f"{job['slug']}.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        job.update(status="done", review=data)
        log("Done")
    except Exception as e:  # report a readable error; never echo keys
        msg = str(e)
        if "429" in msg or "rate limit" in msg.lower():
            msg = "The language model's free rate limit was reached. Try again later, or use your own free Groq key."
        elif "401" in msg:
            msg = "The Groq key was rejected. Check it and try again."
        job.update(status="error", error=msg[:300])
    finally:
        job.pop("_key", None)


def _start(kind: str, label: str, groq_key: str | None, cost: int, target, req) -> dict:
    with lock:
        active = sum(1 for j in jobs.values() if j["status"] == "running")
        if active >= MAX_ACTIVE:
            raise HTTPException(429, "Another run is in progress right now. Please try again in a few minutes.")
        own = (groq_key or "").strip()
        if not own:
            if not _host_quota_ok(cost):
                raise HTTPException(429, f"Today's free runs are used up. Add your own free Groq key to run one now, or come back tomorrow.")
            key = os.getenv("GROQ_API_KEY", "")
            if not key:
                raise HTTPException(503, "This server has no language-model key configured.")
            usage["count"] += cost
        else:
            key = own
        job_id = uuid.uuid4().hex
        jobs[job_id] = {"status": "running", "kind": kind, "question": label, "slug": slug(label),
                        "started": time.time(), "progress": [], "review": None, "paper": None, "error": None}
    threading.Thread(target=target, args=(job_id, req, key), daemon=True).start()
    return {"job": job_id}


@app.post("/api/review")
def start_review(req: ReviewRequest):
    return _start("review", req.question, req.groq_key, 1, _run, req)


def _run_paper(job_id: str, req: PaperRequest, key: str) -> None:
    job = jobs[job_id]

    def log(msg: str) -> None:
        job["progress"].append({"t": round(time.time() - job["started"]), "msg": msg})

    try:
        llm = GroqLLM(api_key=key)
        data = PaperWriter(llm, log=log).run(req.repo).to_dict()
        data["slug"], data["created"] = job["slug"], time.strftime("%Y-%m-%d")
        data["markdown"] = to_markdown(data)
        data["latex"], data["bibtex"] = to_latex(data)
        job.update(status="done", paper=data)  # papers are private to the requester: not added to the gallery
    except Exception as e:
        msg = str(e)
        if "429" in msg or "rate limit" in msg.lower():
            msg = "The language model's free rate limit was reached. Try again later, or use your own free Groq key."
        elif "401" in msg:
            msg = "The Groq key was rejected. Check it and try again."
        job.update(status="error", error=msg[:300])


@app.post("/api/paper")
def start_paper(req: PaperRequest):
    return _start("paper", req.repo.strip().rstrip("/"), req.groq_key, PAPER_COST, _run_paper, req)


@app.get("/api/job/{job_id}")
def job_status(job_id: str):
    j = jobs.get(job_id)
    if not j:
        raise HTTPException(404, "Unknown job")
    return {k: v for k, v in j.items() if not k.startswith("_")}


@app.get("/api/status")
def status():
    _host_quota_ok()
    return {"free_left_today": max(DAILY_CAP - usage["count"], 0), "daily_cap": DAILY_CAP,
            "busy": any(j["status"] == "running" for j in jobs.values())}


@app.get("/api/reviews")
def list_reviews():
    items = []
    for f in sorted(STORE.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:60] if STORE.exists() else []:
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            items.append({"slug": d["slug"], "question": d["question"], "created": d.get("created"), "cards": d["stats"]["cards"]})
        except (json.JSONDecodeError, KeyError):
            continue
    return items


@app.get("/api/reviews/{name}")
def get_review(name: str):
    f = STORE / f"{re.sub(r'[^a-z0-9-]', '', name)}.json"
    if not f.exists():
        raise HTTPException(404, "Not found")
    return JSONResponse(json.loads(f.read_text(encoding="utf-8")))


@app.api_route("/", methods=["GET", "HEAD"])  # Render health checks use HEAD
def home():
    return FileResponse(DOCS / "index.html")


app.mount("/", StaticFiles(directory=DOCS), name="static")
