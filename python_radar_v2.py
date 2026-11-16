#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AI research radar: broad discovery, auditable screening, track-aware reports.

Python >=3.10. Dependencies: requests, feedparser, jinja2.
Works without an LLM key (explicitly marks heuristic classification/summaries).
"""
from __future__ import annotations

import argparse
import html
import json
import logging
import os
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from xml.etree import ElementTree as ET

import requests
from jinja2 import Environment, BaseLoader, select_autoescape

LOG = logging.getLogger("paper_radar")
UTC = timezone.utc
TRACKS = ("rl", "coding_agent", "agentic_rl", "llm_training", "rsi")
TRACK_NAMES = {
    "rl": "🧠 强化学习 (RL)",
    "coding_agent": "💻 Coding Agent",
    "agentic_rl": "🤖 Agentic RL / 多智能体",
    "llm_training": "🏋️ LLM 训练 / 对齐",
    "rsi": "🔄 RSI / 自进化",
    "framework": "🛠️ 框架与基准动态",
}
# Top 12: deterministic track guarantees when enough in-scope candidates exist.
TOP_QUOTAS = {"rsi": 4, "coding_agent": 2, "agentic_rl": 2, "rl": 2, "llm_training": 1}

# Distinct, targeted discovery paths. Query expressions are combined with date limits.
ARXIV_QUERIES = {
    "rsi_recursive": '(ti:"recursive self-improvement" OR abs:"recursive self-improvement" OR abs:"recursive self improvement" OR abs:"godel machine")',
    "rsi_improving": '(ti:"self-improving" OR abs:"self-improving agents" OR abs:"self-improving systems" OR abs:"self improving agent" OR abs:"self-improvement")',
    "rsi_evolving": '(ti:"self-evolving" OR abs:"self-evolving" OR abs:"self-evolution" OR abs:"agent evolution")',
    "rsi_meta": '(abs:"meta-improvement" OR abs:"self-modification" OR abs:"self-modifying" OR abs:"self-optimization" OR abs:"meta-optimization")',
    "rsi_skills": '(abs:"skill evolution" OR abs:"skill-policy" OR abs:"skill acquisition") AND (abs:agent OR abs:LLM OR abs:"language model")',
    "rsi_autoresearch": '(abs:"automated AI research" OR abs:"autonomous research" OR abs:"AI scientist" OR abs:"automated researcher")',
    "rsi_feedback": '(abs:"self-rewarding" OR abs:"self-generated data" OR abs:"self-play" OR abs:"synthetic curriculum") AND (abs:LLM OR abs:agent OR abs:"language model")',
    "rsi_selfrefine": '(ti:"self-refinement" OR abs:"iterative self-refinement" OR abs:"self-debugging" OR abs:"reflection-driven") AND (abs:agent OR abs:"language model")',
    "coding_se": 'cat:cs.SE AND (abs:agent OR abs:"code generation" OR abs:"program repair" OR abs:"issue resolution")',
    "coding_llm": '(cat:cs.CL OR cat:cs.AI OR cat:cs.LG) AND (abs:"coding agent" OR abs:"software engineering" OR abs:"SWE-bench" OR abs:"repository-level")',
    "agentic_agents": '(cat:cs.AI OR cat:cs.CL) AND (abs:"agentic" OR abs:"tool-using agents" OR abs:"multi-agent" OR abs:"agent training")',
    "agentic_policy": '(cat:cs.AI OR cat:cs.LG) AND (abs:"agent reinforcement learning" OR abs:"agentic reinforcement learning" OR abs:"tool use" OR abs:"agent trajectory")',
    "rl_policy": '(cat:cs.LG OR cat:cs.AI OR cat:cs.CL) AND (abs:"reinforcement learning" OR abs:RLHF OR abs:GRPO OR abs:PPO OR abs:"policy optimization")',
    "rl_preference": '(cat:cs.CL OR cat:cs.LG) AND (abs:"direct preference optimization" OR abs:"reward modeling" OR abs:"verifiable rewards" OR abs:"reward model")',
    "training_llm": '(cat:cs.CL OR cat:cs.LG OR cat:cs.AI) AND (abs:"post-training" OR abs:"model alignment" OR abs:"LLM training" OR abs:"test-time scaling")',
}
QUERY_TRACK = {name: ("rsi" if name.startswith("rsi_") else
                      "coding_agent" if name.startswith("coding_") else
                      "agentic_rl" if name.startswith("agentic_") else
                      "rl" if name.startswith("rl_") else "llm_training")
               for name in ARXIV_QUERIES}

GITHUB_REPOS = [
    "volcengine/verl", "verl-project/uni-agent", "modelscope/ms-swift",
    "SWE-agent/SWE-agent", "scaleapi/rsi-benchmark", "quqxui/RSI-Flow",
]

SOFT_EXCLUDE = re.compile(
    r"\b(?:robotics?|robotic|embodied|humanoid|quadruped|locomotion|"
    r"autonomous driving|self-driving|world[- ]models?|world simulator|"
    r"vision-language-action|vla)\b", re.I,
)
TERMS = {
    "rsi_core": ["recursive self-improvement", "recursive self improvement", "godel machine", "gödel machine", "self-modification", "self-modifying"],
    "rsi_related": ["self-improving", "self improving", "self-evolving", "self evolving", "self-evolution", "self evolution", "skill evolution", "co-evolution", "self-rewarding", "self-generated data", "automated ai research", "autonomous research", "self-optimization", "self-improvement", "skill-policy"],
    "rsi_weak": ["self-refine", "self refine", "reflexion", "self-debug", "self debug", "reflection-driven", "self-instruct"],
    "coding_agent": ["coding agent", "code agent", "swe-bench", "software engineering", "repository-level", "repo-level", "program repair", "code generation", "issue resolution"],
    "agentic_rl": ["agentic", "multi-agent", "tool-using agent", "agent training", "agent trajectory", "llm agent", "tool use", "tool-use"],
    "rl": ["reinforcement learning", "rlhf", "grpo", "ppo", "dpo", "policy optimization", "reward model", "reward modeling", "verifiable reward", "self-play"],
    "llm_training": ["llm training", "language model training", "post-training", "post training", "alignment", "fine-tuning", "fine tuning", "pretraining", "test-time scaling", "distillation"],
}
TOKEN_RE = re.compile(r"\b(?:rl|llm|rsi)\b", re.I)
ARXIV_ID_RE = re.compile(r"(?:^|/)(\d{4}\.\d{4,5}|[a-z-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?$", re.I)


@dataclass
class Config:
    discovery_days: int = 21
    report_days: int = 7
    arxiv_page_size: int = 100
    arxiv_max_pages: int = 5
    arxiv_delay_seconds: float = 3.1
    hf_page_size: int = 100
    hf_max_pages_per_week: int = 8
    github_releases_per_repo: int = 6
    max_llm_candidates: int = 500  # overflow gets explicitly marked heuristic scores
    batch_size: int = 20
    top_count: int = 12
    worth_per_track: int = 10
    state_retention_days: int = 45
    http_timeout: int = 35
    http_retries: int = 3
    out_dir: Path = Path(".")
    llm_api_key: str = ""
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_model: str = "deepseek-chat"
    github_token: str = ""
    hf_token: str = ""
    webhook_url: str = ""
    use_llm: bool = True

    def __post_init__(self):
        self.out_dir = Path(self.out_dir)
        if self.discovery_days < self.report_days or self.report_days < 1:
            raise ValueError("discovery-days must be >= report-days >= 1")
        if self.batch_size < 1 or self.top_count < 1 or self.arxiv_page_size < 1:
            raise ValueError("invalid batch/page/top parameters")


def now_utc() -> datetime:
    return datetime.now(UTC)


def to_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)
    except (ValueError, TypeError):
        try:
            return datetime.strptime(str(value)[:10], "%Y-%m-%d").replace(tzinfo=UTC)
        except ValueError:
            return None


def recent(value: str | None, days: int, clock: datetime) -> bool:
    dt = to_dt(value)
    return dt is not None and clock - timedelta(days=days) <= dt <= clock + timedelta(days=1)


def normalize_title(value: str) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]", "", value.casefold())[:160]


def clean_arxiv_id(value: str | None) -> str:
    if not value:
        return ""
    value = str(value).strip().rstrip("/")
    if "/abs/" in value:
        value = value.rsplit("/abs/", 1)[-1]
    match = ARXIV_ID_RE.search(value)
    if match:
        return match.group(1)
    # HuggingFace /papers/{arxiv_id} sometimes contains a plain id.
    match = re.fullmatch(r"\d{4}\.\d{4,5}(?:v\d+)?", value)
    return re.sub(r"v\d+$", "", value) if match else ""


def paper_key(p: dict) -> str:
    arxiv_id = clean_arxiv_id(p.get("arxiv_id"))
    return f"arxiv:{arxiv_id}" if arxiv_id else "title:" + normalize_title(p.get("title", ""))


def safe_url(url: str) -> str:
    try:
        parsed = urlparse(str(url))
        return str(url) if parsed.scheme in {"https", "http"} and parsed.netloc else "#"
    except ValueError:
        return "#"


def term_match(text: str, word: str) -> bool:
    return bool(re.search(r"(?<!\w)" + re.escape(word) + r"(?!\w)", text, re.I))


def keyword_hits(text: str, words: list[str]) -> int:
    return sum(term_match(text, w) for w in words)


def heuristic_track(p: dict) -> tuple[str, str, dict]:
    """Cheap broad classifier; RSI levels are discovery leads, NOT proof of recursion."""
    if p.get("source") == "github_release" or "github_release" in p.get("sources", []):
        return "framework", "none", {}
    content = f'{p.get("title", "")} {p.get("abstract", "")}'.lower()
    scores = {t: keyword_hits(content, TERMS[t]) for t in TRACKS if t != "rsi"}
    core = keyword_hits(content, TERMS["rsi_core"])
    related = keyword_hits(content, TERMS["rsi_related"])
    weak = keyword_hits(content, TERMS["rsi_weak"])
    rsi_level = "core_candidate" if core else "related" if related else "weak" if weak else "none"
    scores["rsi"] = core * 5 + related * 3 + weak
    # Full-text matches allow HF additions that mention the niche only in the abstract.
    if term_match(content, "rl"):
        scores["rl"] += 1
    if term_match(content, "llm"):
        scores["llm_training"] += 1
    query_tracks = {QUERY_TRACK[q] for q in p.get("matched_queries", []) if q in QUERY_TRACK}
    for t in query_tracks:
        scores[t] = scores.get(t, 0) + 2
    # Prefer RSI if its distinctive signals exist; RL self-play alone is not RSI.
    best = max(TRACKS, key=lambda t: (scores.get(t, 0), t == "rsi"))
    if scores.get(best, 0) == 0:
        best = "llm_training"
    return best, rsi_level, scores


def heuristic_score(p: dict, clock: datetime) -> dict:
    track, rsi_level, scores = heuristic_track(p)
    content = f'{p.get("title", "")} {p.get("abstract", "")}'
    # Never infer authoritative institutions or experiment results from the abstract.
    relevance = min(40, 16 + 5 * min(scores.get(track, 0), 5))
    depth = min(25, 8 + (6 if "method" in content.lower() else 0) +
                (6 if "algorithm" in content.lower() or "framework" in content.lower() else 0))
    evidence = min(20, 6 + (5 if re.search(r"\b(?:experiment|evaluation|benchmark)\b", content, re.I) else 0) +
                   (4 if re.search(r"\b(?:results?|outperform|improve)\b", content, re.I) else 0))
    dt = to_dt(p.get("activity_date"))
    age = (clock - dt).days if dt else 30
    freshness = 15 if age <= 3 else 11 if age <= 7 else 6 if age <= 14 else 2
    # Target-matching is not equivalent to main-topic relevance; model may reject.
    primary_title = p.get("title", "")
    scope = "review" if SOFT_EXCLUDE.search(primary_title) else "in_scope"
    return {"track": track, "rsi_level": rsi_level, "in_scope": scope,
            "score_breakdown": {"relevance": relevance, "technical_depth": depth,
                                "evidence": evidence, "timeliness": freshness},
            "total_score": relevance + depth + evidence + freshness,
            "reason": "启发式候选评分（未经过模型内容核查）", "score_origin": "heuristic"}


class HttpClient:
    def __init__(self, config: Config):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "PaperRadarV2/2.0 (academic weekly digest; polite API client)",
                                     "Accept": "application/json, application/atom+xml, */*"})
        self._last_arxiv_request: float | None = None

    def request(self, method: str, url: str, *, params=None, json_body=None, headers=None,
                timeout: int | None = None) -> requests.Response:
        last_exception = None
        for attempt in range(self.config.http_retries):
            if "export.arxiv.org" in url:
                elapsed = (time.monotonic() - self._last_arxiv_request) if self._last_arxiv_request is not None else 1e9
                wait = max(0, self.config.arxiv_delay_seconds - elapsed)
                if wait:
                    time.sleep(wait)
                self._last_arxiv_request = time.monotonic()
            try:
                response = self.session.request(method, url, params=params, json=json_body,
                                                headers=headers, timeout=timeout or self.config.http_timeout)
                if response.status_code in {429, 500, 502, 503, 504}:
                    retry_after = response.headers.get("Retry-After", "")
                    delay = min(60, int(retry_after)) if retry_after.isdigit() else min(32, 2 ** attempt)
                    last_exception = RuntimeError(f"HTTP {response.status_code} {url}")
                    if attempt < self.config.http_retries - 1:
                        LOG.warning("Retry %s: %s; %.1fs", attempt + 1, last_exception, delay)
                        time.sleep(delay)
                        continue
                response.raise_for_status()
                return response
            except (requests.RequestException, RuntimeError) as e:
                last_exception = e
                if attempt < self.config.http_retries - 1:
                    time.sleep(min(32, 2 ** attempt))
        raise RuntimeError(f"Request failed: {url}: {last_exception}")


def parse_arxiv_atom(xml_data: bytes) -> list[dict]:
    """Parse arXiv Atom using the standard library (no feedparser dependency)."""
    root = ET.fromstring(xml_data)
    atom = "{http://www.w3.org/2005/Atom}"
    parsed = []
    for entry in root.findall(f"{atom}entry"):
        val = lambda field: (entry.findtext(f"{atom}{field}") or "").strip()
        authors = [{"name": (node.findtext(f"{atom}name") or "").strip()}
                   for node in entry.findall(f"{atom}author")]
        categories = [{"term": node.attrib.get("term", "")}
                      for node in entry.findall(f"{atom}category")]
        parsed.append({"id": val("id"), "title": val("title"), "summary": val("summary"),
                       "published": val("published"), "updated": val("updated"),
                       "authors": authors, "tags": categories})
    return parsed


class Fetcher:
    def __init__(self, config: Config, http: HttpClient, clock: datetime):
        self.c, self.http, self.clock = config, http, clock
        self.metrics = Counter()
        self.per_query: dict[str, dict] = {}

    def fetch_arxiv(self) -> list[dict]:
        papers = []
        start_utc = self.clock - timedelta(days=self.c.discovery_days)
        begin = start_utc.strftime("%Y%m%d%H%M")
        end = self.clock.strftime("%Y%m%d%H%M")
        for name, expression in ARXIV_QUERIES.items():
            query = f"({expression}) AND submittedDate:[{begin} TO {end}]"
            result_count = 0
            error = ""
            for page in range(self.c.arxiv_max_pages):
                params = {"search_query": query, "start": page * self.c.arxiv_page_size,
                          "max_results": self.c.arxiv_page_size,
                          "sortBy": "submittedDate", "sortOrder": "descending"}
                try:
                    response = self.http.request("GET", "https://export.arxiv.org/api/query", params=params)
                    entries = parse_arxiv_atom(response.content)
                except Exception as exc:
                    error = str(exc)
                    self.metrics["arxiv_query_errors"] += 1
                    LOG.warning("arXiv %s page=%s: %s", name, page + 1, exc)
                    break
                self.metrics["arxiv_raw_results"] += len(entries)
                result_count += len(entries)
                for entry in entries:
                    aid = clean_arxiv_id(entry.get("id", ""))
                    if not aid:
                        self.metrics["arxiv_invalid_id"] += 1
                        continue
                    title = " ".join(entry.get("title", "").split())
                    abstract = " ".join(entry.get("summary", "").split())
                    published = entry.get("published", "")
                    if not recent(published, self.c.discovery_days, self.clock):
                        self.metrics["arxiv_date_excluded"] += 1
                        continue
                    authors = [a.get("name", "") for a in entry.get("authors", []) if a.get("name")]
                    tags = [t.get("term", "") for t in entry.get("tags", []) if t.get("term")]
                    paper = {"source": "arxiv_query", "sources": ["arxiv_query"],
                             "arxiv_id": aid, "title": title,
                             "url": f"https://arxiv.org/abs/{aid}",
                             "authors": ", ".join(authors[:5]) + (" et al." if len(authors) > 5 else ""),
                             "abstract": abstract, "published": published[:10],
                             "updated": entry.get("updated", "")[:10],
                             "activity_date": published[:10], "categories": tags,
                             "matched_queries": [name],
                             "discovery_tracks": [QUERY_TRACK[name]],
                             "soft_exclusion": bool(SOFT_EXCLUDE.search(title + " " + abstract))}
                    papers.append(paper)
                if len(entries) < self.c.arxiv_page_size:
                    break
            self.per_query[name] = {"raw_results": result_count, "error": error,
                                    "capped": result_count >= self.c.arxiv_max_pages * self.c.arxiv_page_size}
            LOG.info("arXiv %-18s %4d records%s", name, result_count,
                     " [CAP]" if self.per_query[name]["capped"] else "")
        return papers

    def fetch_hf(self) -> list[dict]:
        results = []
        # Query the weeks intersecting our discovery window; HF's API supports `week`, p, limit.
        start_date = (self.clock - timedelta(days=self.c.discovery_days)).date()
        end_date = self.clock.date()
        weeks = sorted({(start_date + timedelta(days=i)).isocalendar()[:2]
                        for i in range((end_date - start_date).days + 1)})
        headers = {"Authorization": f"Bearer {self.c.hf_token}"} if self.c.hf_token else None
        for year, week in weeks:
            week_str = f"{year}-W{week:02d}"
            raw_count, error = 0, ""
            for page in range(self.c.hf_max_pages_per_week):
                try:
                    response = self.http.request("GET", "https://huggingface.co/api/daily_papers",
                                                 params={"week": week_str, "p": page,
                                                         "limit": self.c.hf_page_size, "sort": "publishedAt"},
                                                 headers=headers)
                    items = response.json()
                    if not isinstance(items, list):
                        raise ValueError("HF response is not a JSON list")
                except Exception as exc:
                    error = str(exc)
                    self.metrics["hf_query_errors"] += 1
                    LOG.warning("HF week %s page %d: %s", week_str, page, exc)
                    break
                raw_count += len(items)
                self.metrics["hf_raw_results"] += len(items)
                for item in items:
                    p = item.get("paper") or {}
                    if not isinstance(p, dict):
                        continue
                    title = (p.get("title") or "").strip()
                    summary = (p.get("summary") or "").strip()
                    if not title:
                        continue
                    pub = p.get("publishedAt") or ""
                    daily = item.get("submittedOnDailyAt") or p.get("submittedOnDailyAt") or ""
                    activity = daily or pub
                    if not recent(activity, self.c.discovery_days, self.clock):
                        self.metrics["hf_date_excluded"] += 1
                        continue
                    aid = clean_arxiv_id(p.get("id") or "")
                    authors = p.get("authors") or []
                    names = [x.get("name", "") if isinstance(x, dict) else str(x) for x in authors]
                    rec = {"source": "hf_daily", "sources": ["hf_daily"],
                           "arxiv_id": aid, "title": title, "abstract": summary,
                           "url": f"https://huggingface.co/papers/{aid}" if aid else p.get("url", ""),
                           "authors": ", ".join(names[:5]) + (" et al." if len(names) > 5 else ""),
                           "published": pub[:10], "activity_date": activity[:10],
                           "hf_featured_at": daily[:10], "categories": [],
                           "matched_queries": ["hf_week:" + week_str], "discovery_tracks": [],
                           "soft_exclusion": bool(SOFT_EXCLUDE.search(title + " " + summary)),
                           "hf_upvotes": item.get("numUpvotes", 0)}
                    track, rsi, scores = heuristic_track(rec)
                    # HF is supplementary; do not gate RSI behind RL keywords.
                    if max(scores.values(), default=0) <= 1 and rsi == "none":
                        self.metrics["hf_low_relevance"] += 1
                        continue
                    results.append(rec)
                if len(items) < self.c.hf_page_size:
                    break
            self.per_query["hf:" + week_str] = {"raw_results": raw_count, "error": error,
                                                  "capped": raw_count >= self.c.hf_max_pages_per_week * self.c.hf_page_size}
        return results

    def fetch_github(self) -> list[dict]:
        results = []
        headers = {"Authorization": f"Bearer {self.c.github_token}"} if self.c.github_token else None
        for repo in GITHUB_REPOS:
            try:
                response = self.http.request("GET", f"https://api.github.com/repos/{repo}/releases",
                                             params={"per_page": self.c.github_releases_per_repo}, headers=headers)
                releases = response.json()
                if not isinstance(releases, list):
                    raise ValueError("GitHub releases was not a list")
            except Exception as exc:
                self.metrics["github_errors"] += 1
                self.per_query["github:" + repo] = {"error": str(exc)}
                LOG.warning("GitHub %s: %s", repo, exc)
                continue
            self.per_query["github:" + repo] = {"raw_results": len(releases), "error": ""}
            self.metrics["github_raw_results"] += len(releases)
            for release in releases:
                published = release.get("published_at") or ""
                if not recent(published, self.c.discovery_days, self.clock):
                    continue
                body = re.sub(r"<[^>]+>", "", release.get("body") or "")[:5000]
                results.append({"source": "github_release", "sources": ["github_release"],
                                "arxiv_id": "", "title": f"[Framework Update] {repo}: {release.get('name') or release.get('tag_name') or 'Release'}",
                                "url": release.get("html_url", f"https://github.com/{repo}"),
                                "authors": repo, "abstract": body, "published": published[:10],
                                "activity_date": published[:10], "categories": ["Framework", "GitHub"],
                                "matched_queries": ["github:" + repo], "discovery_tracks": ["framework"],
                                "soft_exclusion": False})
        return results

    def fetch_all(self) -> list[dict]:
        a = self.fetch_arxiv()
        h = self.fetch_hf()
        g = self.fetch_github()
        LOG.info("Fetched rows arXiv=%d HF=%d GitHub=%d", len(a), len(h), len(g))
        return a + h + g


def merge_records(records: list[dict]) -> list[dict]:
    """Keep original arXiv URL/abstract when HF duplicates it; retain all provenance."""
    merged: dict[str, dict] = {}
    for p in records:
        if not p.get("title"):
            continue
        k = paper_key(p)
        if k not in merged:
            merged[k] = dict(p)
            merged[k]["sources"] = sorted(set(p.get("sources", [p.get("source", "unknown")])))
            merged[k]["matched_queries"] = sorted(set(p.get("matched_queries", [])))
            merged[k]["discovery_tracks"] = sorted(set(p.get("discovery_tracks", [])))
            continue
        existing = merged[k]
        for field in ("sources", "matched_queries", "discovery_tracks", "categories"):
            existing[field] = sorted(set(existing.get(field, [])) | set(p.get(field, [])))
        existing["soft_exclusion"] = bool(existing.get("soft_exclusion") or p.get("soft_exclusion"))
        if (to_dt(p.get("activity_date")) or datetime.min.replace(tzinfo=UTC)) > (to_dt(existing.get("activity_date")) or datetime.min.replace(tzinfo=UTC)):
            existing["activity_date"] = p["activity_date"]
        if p.get("hf_featured_at"):
            existing["hf_featured_at"] = p["hf_featured_at"]
        existing["hf_upvotes"] = max(existing.get("hf_upvotes", 0) or 0, p.get("hf_upvotes", 0) or 0)
        if p.get("source") == "arxiv_query" and existing.get("source") != "arxiv_query":
            for field in ("source", "title", "url", "authors", "abstract", "published", "updated"):
                if p.get(field):
                    existing[field] = p[field]
        elif len(p.get("abstract", "")) > len(existing.get("abstract", "")) and existing.get("source") != "arxiv_query":
            existing["abstract"] = p["abstract"]
        if not existing.get("arxiv_id") and p.get("arxiv_id"):
            existing["arxiv_id"] = p["arxiv_id"]
    return list(merged.values())


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        LOG.warning("Cannot read %s: %s", path, exc)
        return default


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def parse_json_object(text: str | None) -> dict | None:
    if not text:
        return None
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    try:
        value = json.loads(cleaned)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        try:
            start, end = cleaned.index("{"), cleaned.rindex("}") + 1
            value = json.loads(cleaned[start:end])
            return value if isinstance(value, dict) else None
        except (ValueError, json.JSONDecodeError):
            return None


class LLM:
    def __init__(self, config: Config, http: HttpClient):
        self.c, self.http = config, http
        self.enabled = config.use_llm and bool(config.llm_api_key)
        if not self.enabled:
            LOG.warning("No LLM enabled/key; using clearly labeled heuristic fallback")

    def complete(self, prompt: str, max_tokens: int = 4500) -> dict | None:
        if not self.enabled:
            return None
        url = self.c.llm_base_url.rstrip("/")
        if not url.endswith("/chat/completions"):
            url += "/chat/completions"
        try:
            response = self.http.request("POST", url, headers={
                "Authorization": f"Bearer {self.c.llm_api_key}",
                "Content-Type": "application/json",
            }, timeout=180, json_body={"model": self.c.llm_model,
                          "messages": [{"role": "system", "content": "你是严格依据摘要判断 AI 论文的研究助理。只能返回合法 JSON。"},
                                       {"role": "user", "content": prompt}],
                          "temperature": 0.1, "max_tokens": max_tokens})
            data = response.json()
            msg = data["choices"][0]["message"]["content"]
            parsed = parse_json_object(msg)
            if parsed is None:
                LOG.warning("LLM response could not be parsed: %s", str(msg)[-250:])
            return parsed
        except Exception as exc:
            LOG.warning("LLM request failed: %s", exc)
            return None


def validate_llm_score(item: dict, clock: datetime, paper: dict) -> dict | None:
    if not isinstance(item, dict):
        return None
    track = item.get("track")
    if track not in TRACKS:
        return None
    rsi_level = item.get("rsi_level", "none")
    if rsi_level not in {"core", "related", "weak", "none"}:
        rsi_level = "none"
    scope = item.get("in_scope", "in_scope")
    if scope not in {"in_scope", "out_of_scope", "uncertain"}:
        scope = "uncertain"
    score = item.get("score_breakdown") or {}
    maxima = {"relevance": 40, "technical_depth": 25, "evidence": 20, "timeliness": 15}
    breakdown = {}
    for k, bound in maxima.items():
        v = score.get(k)
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            return None
        breakdown[k] = min(bound, max(0, int(v)))
    return {"track": track, "rsi_level": rsi_level, "in_scope": scope,
            "score_breakdown": breakdown, "total_score": sum(breakdown.values()),
            "reason": str(item.get("reason", ""))[:260], "score_origin": "llm"}


def choose_for_llm(papers: list[dict], limit: int, clock: datetime) -> set[str]:
    """Stratified, RSI-favoring candidate budget: remaining candidates stay available."""
    if limit <= 0:
        return set()
    bins = defaultdict(list)
    for p in papers:
        bins[p["assessment"]["track"]].append(p)
    for items in bins.values():
        items.sort(key=lambda p: (p["assessment"]["total_score"], p.get("activity_date", "")), reverse=True)
    # Reserve up to half the LLM budget for RSI if that pool is large.
    take = []
    rsi_limit = max(1, limit // 2)
    take.extend(bins["rsi"][:rsi_limit])
    seen = {paper_key(p) for p in take}
    # Round-robin across all tracks to protect breadth.
    positions = {track: (len(take) if track == "rsi" else 0) for track in TRACKS}
    while len(take) < limit:
        advanced = False
        for track in TRACKS:
            items = bins[track]
            idx = positions[track]
            if idx < len(items) and len(take) < limit:
                positions[track] += 1
                candidate = items[idx]
                if paper_key(candidate) not in seen:
                    take.append(candidate)
                    seen.add(paper_key(candidate))
                    advanced = True
        if not advanced:
            break
    return seen


def score_candidates(papers: list[dict], c: Config, clock: datetime, llm: LLM) -> Counter:
    stats = Counter()
    for p in papers:
        p["assessment"] = heuristic_score(p, clock)
    pending = [p for p in papers if p.get("source") != "github_release"]
    selected_keys = choose_for_llm(pending, c.max_llm_candidates, clock)
    selected = [p for p in pending if paper_key(p) in selected_keys]
    stats["llm_eligible"] = len(selected)
    stats["llm_overflow_heuristic"] = len(pending) - len(selected)
    for offset in range(0, len(selected), c.batch_size):
        batch = selected[offset:offset + c.batch_size]
        data = [{"id": str(i), "title": p["title"], "abstract": p.get("abstract", "")[:1400],
                 "categories": p.get("categories", []), "matched_queries": p.get("matched_queries", []),
                 "soft_exclusion": p.get("soft_exclusion", False),
                 "published": p.get("activity_date", "")}
                for i, p in enumerate(batch)]
        prompt = (
            f"当前 UTC 时间为 {clock.date().isoformat()}。"
            "对这些 AI 论文逐篇评分/分类。请使用摘要证据，而不是单靠词语命中或猜测机构。"
            "主赛道 track 只能为 rl,coding_agent,agentic_rl,llm_training,rsi。"
            "RSI rsi_level: core=证据明确表明递归改进机制本身，related=持久技能/模型/工作流自我改进但未证实递归，"
            "weak=单任务自修正，none=无关。并非所有 self-play 都是 RSI。"
            "in_scope: in_scope/out_of_scope/uncertain。只在机器人/具身/世界模型是主要贡献时 out_of_scope；"
            "摘要提及这些词但主要方法为通用 AI 则不要拒绝。"
            "四维整数分数：relevance 0..40,technical_depth 0..25,evidence 0..20,timeliness 0..15。"
            "请按每个 id 返回一条，禁止臆造实验指标。仅返回 JSON: "
            '{"scores":[{"id":"0","track":"rsi","rsi_level":"related","in_scope":"in_scope",'
            '"score_breakdown":{"relevance":35,"technical_depth":20,"evidence":10,"timeliness":15},'
            '"reason":"解释分类依据"}]}\n'
            "输入：" + json.dumps(data, ensure_ascii=False)
        )
        result = llm.complete(prompt, max_tokens=max(2500, 240 * len(batch) + 500))
        mapping = {}
        if isinstance(result, dict):
            for row in result.get("scores", []):
                if isinstance(row, dict):
                    mapping[str(row.get("id", ""))] = row
        for i, p in enumerate(batch):
            valid = validate_llm_score(mapping.get(str(i), {}), clock, p)
            if valid:
                p["assessment"] = valid
                stats["llm_scored"] += 1
            else:
                stats["heuristic_fallback"] += 1
    stats["total_scored"] = len(pending)
    return stats


def select_for_report(papers: list[dict], c: Config, clock: datetime) -> tuple[dict, list]:
    """Deterministic selection. `top_papers` never lets one large track hide RSI."""
    eligible = [p for p in papers if p.get("source") != "github_release" and
                recent(p.get("activity_date"), c.report_days, clock) and
                p.get("assessment", {}).get("in_scope") != "out_of_scope"]
    grouped = defaultdict(list)
    for p in eligible:
        grouped[p["assessment"]["track"]].append(p)
    for t in TRACKS:
        grouped[t].sort(key=lambda p: (p["assessment"]["total_score"],
                                        p.get("activity_date", "")), reverse=True)
    def eligible_for_feature(p: dict) -> bool:
        assessment = p["assessment"]
        # Single-shot self-refinement can be listed as related reading, not strong RSI.
        return not (assessment["track"] == "rsi" and assessment.get("rsi_level") in {"weak", "none"})

    top = []
    chosen = set()
    for track, quota in TOP_QUOTAS.items():
        featureable = [p for p in grouped[track] if eligible_for_feature(p)]
        for p in featureable[:min(quota, c.top_count - len(top))]:
            key = paper_key(p)
            if key not in chosen:
                top.append(p)
                chosen.add(key)
    all_remaining = sorted((p for p in eligible if paper_key(p) not in chosen and eligible_for_feature(p)),
                           key=lambda p: (p["assessment"]["total_score"], p.get("activity_date", "")), reverse=True)
    for p in all_remaining[:max(0, c.top_count - len(top))]:
        top.append(p)
        chosen.add(paper_key(p))
    top.sort(key=lambda p: (p["assessment"]["total_score"], p.get("activity_date", "")), reverse=True)
    selections = {}
    for track in TRACKS:
        in_top = [p for p in top if p["assessment"]["track"] == track]
        worth = [p for p in grouped[track] if paper_key(p) not in chosen][:c.worth_per_track]
        selections[track] = {"recommended": in_top, "worth_knowing": worth,
                             "available": len(grouped[track])}
    gh = [p for p in papers if p.get("source") == "github_release" and recent(p.get("activity_date"), c.report_days, clock)]
    gh.sort(key=lambda p: p.get("published", ""), reverse=True)
    selections["framework"] = {"recommended": gh, "worth_knowing": [], "available": len(gh)}
    return selections, top


def summarize_papers(selections: dict, top: list, llm: LLM, c: Config) -> dict[str, dict]:
    """The LLM cannot alter IDs, titles, paper selections or links."""
    order = []
    visited = set()
    for t in [*TRACKS, "framework"]:
        for group in ["recommended", "worth_knowing"]:
            for p in selections[t][group]:
                key = paper_key(p)
                if key not in visited:
                    order.append(p)
                    visited.add(key)
    content = {}
    for start in range(0, len(order), 8):
        batch = order[start:start + 8]
        payload = [{"id": str(i), "title": p["title"], "abstract": p.get("abstract", "")[:1800]}
                   for i, p in enumerate(batch)]
        prompt = (
            "只依据给出的摘要/Release Note 做中文简述，严禁编造指标、实验结果、公开代码情况。"
            "针对每项输出 core_contribution(<=130字),technical_highlights(<=90字),"
            "differentiation(<=70字),application_direction(<=90字),one_line_summary(<=75字),"
            "highlights(<=80字),application_tip(<=80字)。没有证据的字段写'摘要未说明'。"
            '仅返回 JSON 格式 {"items":[{"id":"0","core_contribution":"...",'
            '"technical_highlights":"...","differentiation":"...",'
            '"application_direction":"...","one_line_summary":"...",'
            '"highlights":"...","application_tip":"..."}]}。\n'
            "论文：" + json.dumps(payload, ensure_ascii=False)
        )
        result = llm.complete(prompt, max_tokens=3200)
        by_id = {}
        if isinstance(result, dict):
            for row in result.get("items", []):
                if isinstance(row, dict):
                    by_id[str(row.get("id", ""))] = row
        for i, p in enumerate(batch):
            row = by_id.get(str(i), {})
            fallback = (p.get("abstract") or "摘要暂缺")[:380]
            content[paper_key(p)] = {
                "core_contribution": str(row.get("core_contribution") or fallback)[:420],
                "technical_highlights": str(row.get("technical_highlights") or "摘要未说明")[:250],
                "differentiation": str(row.get("differentiation") or "摘要未说明")[:200],
                "application_direction": str(row.get("application_direction") or "需要阅读全文评估")[:250],
                "one_line_summary": str(row.get("one_line_summary") or fallback[:140])[:220],
                "highlights": str(row.get("highlights") or "摘要未说明")[:200],
                "application_tip": str(row.get("application_tip") or "需要阅读全文评估")[:200],
                "summary_origin": "llm" if row else "source_excerpt",
            }
    return content


def build_report(selections: dict, top: list, summaries: dict, clock: datetime, stats: dict) -> dict:
    def item(p: dict) -> dict:
        assessment = p.get("assessment", {})
        data = {"id": paper_key(p), "title": p.get("title", ""), "url": safe_url(p.get("url", "")),
                "authors": p.get("authors") or "N/A", "published": p.get("published") or "N/A",
                "activity_date": p.get("activity_date", ""), "categories": p.get("categories", []),
                "source": p.get("source", ""), "sources": p.get("sources", []),
                "rsi_level": assessment.get("rsi_level", "none"),
                "score_origin": assessment.get("score_origin", "unscored"),
                "in_scope": assessment.get("in_scope", "uncertain"),
                "total_score": assessment.get("total_score", "N/A"),
                "score_breakdown": assessment.get("score_breakdown", {}),
                "reason": assessment.get("reason", ""),
                "soft_exclusion": p.get("soft_exclusion", False),
                "abstract": p.get("abstract", ""),
                "code_availability": "未核查", "difficulty": "未评估"}
        data.update(summaries.get(paper_key(p), {}))
        return data

    tracks = []
    for track in [*TRACKS, "framework"]:
        group = selections[track]
        tracks.append({"track_id": track, "track_name": TRACK_NAMES[track],
                       "available": group["available"],
                       "recommended": [item(p) for p in group["recommended"]],
                       "worth_knowing": [item(p) for p in group["worth_knowing"]]})
    summary = {track: f"本周识别 {selections[track]['available']} 篇相关候选，"
                      f"精选 {len(selections[track]['recommended'])} 篇；"
                      f"另列 {len(selections[track]['worth_knowing'])} 篇值得知道。"
               for track in TRACKS}
    summary["rsi_self_evolution"] = summary.pop("rsi")
    summary["framework_updates"] = f"本周发现 {selections['framework']['available']} 条仓库 Release 更新。"
    return {"report_date": clock.date().isoformat(), "window_days": stats.get("report_days", 7),
            "executive_summary": summary, "cross_insights": [], "paper_tracks": tracks,
            "top_papers": [{"title": p["title"], "url": safe_url(p.get("url", "")),
                            "reason": p["assessment"].get("reason", ""),
                            "track": p["assessment"].get("track", "")} for p in top],
            "diagnostics": stats}


def enrich_editorial(report: dict, selections: dict, top: list, llm: LLM) -> None:
    """Optional short editorial pass; hard-coded article metadata remains unchanged."""
    data = [{"track": p["assessment"]["track"], "title": p["title"],
             "abstract": p.get("abstract", "")[:650]}
            for p in top]
    data.extend({"track": "framework", "title": p["title"],
                 "abstract": p.get("abstract", "")[:650]}
                for p in selections["framework"]["recommended"][:5])
    prompt = (
        "严格只依据以下论文摘要，为周报写六段执行摘要以及最多三条跨领域洞察。"
        "不要引入摘要外的数字或结论；没有合适的跨领域洞察则返回空列表。"
        "字段仅使用 rl,coding_agent,agentic_rl,llm_training,rsi_self_evolution,framework_updates。"
        "每条执行摘要不超过130字，洞察标题<=30字、内容<=160字。"
        '请仅输出 JSON: {"executive_summary":{"rl":"...","coding_agent":"...",'
        '"agentic_rl":"...","llm_training":"...","rsi_self_evolution":"...",'
        '"framework_updates":"..."},"cross_insights":[{"title":"...","content":"..."}]}。'
        "论文数据：" + json.dumps(data, ensure_ascii=False)
    )
    result = llm.complete(prompt, max_tokens=1900)
    if not isinstance(result, dict):
        return
    provided = result.get("executive_summary", {})
    if isinstance(provided, dict):
        for key in report["executive_summary"]:
            value = provided.get(key)
            if isinstance(value, str) and value.strip():
                # Do not substitute fabricated updates when no GitHub release was found.
                if key == "framework_updates" and selections["framework"]["available"] == 0:
                    continue
                report["executive_summary"][key] = value.strip()[:250]
    insights = result.get("cross_insights", [])
    if isinstance(insights, list):
        report["cross_insights"] = [
            {"title": row["title"][:100], "content": row["content"][:350]}
            for row in insights[:3]
            if isinstance(row, dict) and isinstance(row.get("title"), str)
            and isinstance(row.get("content"), str) and row["title"].strip() and row["content"].strip()
        ]


def report_markdown(report: dict) -> str:
    # Escape dynamically sourced strings to avoid raw HTML injection in Markdown.
    e = lambda x: html.escape(str(x or ""), quote=False).replace("\n", " ")
    def link(label: str, url: str) -> str:
        safe_label = e(label).replace("[", "\\[").replace("]", "\\]")
        return f"[{safe_label}]({safe_url(url)})"
    lines = [f"# 🤖 AI 前沿周报 | {report['report_date']}", "",
             f"发现窗口：最近 {report['diagnostics'].get('discovery_days', 21)} 天；"
             f"周报窗口：最近 {report['window_days']} 天。", "",
             "## 📌 执行摘要", ""]
    for track in report["paper_tracks"]:
        summary_key = "rsi_self_evolution" if track["track_id"] == "rsi" else "framework_updates" if track["track_id"] == "framework" else track["track_id"]
        lines.append(f"- **{track['track_name']}**：{e(report['executive_summary'].get(summary_key, ''))}")
    lines.extend(["", "**说明**：启发式评分/原文摘要的条目会明确标注，不等于已阅读论文全文。", ""])
    if report.get("cross_insights"):
        lines.extend(["## 💡 跨领域洞察", ""])
        for insight in report["cross_insights"]:
            lines.extend([f"### {e(insight['title'])}", "", e(insight["content"]), ""])
    for track in report["paper_tracks"]:
        lines.extend([f"## {track['track_name']}（本周候选 {track['available']}）", ""])
        for category, label in (("recommended", "🌟 精选"), ("worth_knowing", "📖 值得知道")):
            entries = track[category]
            if not entries:
                continue
            lines.extend([f"### {label}", ""])
            for p in entries:
                lines.extend([f"#### {link(p['title'], p['url'])}", "",
                              f"作者：{e(p['authors'])} · 发表：{e(p['published'])} · "
                              f"评分：{e(p['total_score'])}（{e(p['score_origin'])}）"
                              + (f" · RSI={e(p['rsi_level'])}" if track["track_id"] == "rsi" else ""), "",
                              e(p.get("core_contribution") or p.get("abstract", "")[:260]), ""])
                if category == "recommended":
                    lines.extend([f"技术亮点：{e(p.get('technical_highlights', ''))}", "",
                                  f"应用方向：{e(p.get('application_direction', ''))}", ""])
                if p.get("soft_exclusion"):
                    lines.extend(["*提示：摘要包含具身/机器人/世界模型相关词，需人工复核主要贡献。*", ""])
    lines.extend(["## 🏆 本周重点阅读", ""])
    for index, p in enumerate(report["top_papers"], 1):
        lines.append(f"{index}. {link(p['title'], p['url'])} — {e(p.get('track', ''))}")
    lines.extend(["", "## 📊 获取与筛选统计", "", "```json",
                  json.dumps(report["diagnostics"], ensure_ascii=False, indent=2), "```", ""])
    return "\n".join(lines)


HTML_TEMPLATE = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI 前沿周报 · {{ report.report_date }}</title>
<style>
:root{color-scheme:light;--ink:#17243a;--muted:#637086;--line:#dce3ed;--bg:#f6f8fb;--card:white;--accent:#2155ab}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.65 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}
main{max-width:1080px;margin:35px auto;padding:0 18px}h1{font-size:2.1rem;margin-bottom:.3rem}h2{font-size:1.5rem;margin-top:42px}
h3{font-size:1.1rem;margin-top:8px}a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}
.card{background:var(--card);border:1px solid var(--line);border-radius:13px;padding:19px 22px;margin:14px 0}
.meta,.muted{color:var(--muted);font-size:.92rem}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}
.kpi{background:#eef3fb;border-radius:10px;padding:14px}.kpi strong{display:block;font-size:1.4rem}.tag{font-size:.76rem;background:#edf1f8;border-radius:999px;padding:2px 8px;display:inline-block;margin-right:5px}
small{color:var(--muted)}details{margin-top:12px}summary{cursor:pointer}pre{overflow:auto;font-size:12px}
</style></head><body><main>
<header><h1>🤖 AI 前沿周报</h1><div class="meta">{{ report.report_date }} · {{ report.window_days }} 天周报窗口 · {{ report.diagnostics.discovery_days }} 天发现窗口</div></header>
<section class="grid">
{% for track in report.paper_tracks %}<div class="kpi"><span>{{ track.track_name }}</span><strong>{{ track.available }}</strong><small>候选 · 精选 {{ track.recommended|length }} 篇</small></div>{% endfor %}
</section><div class="card"><h3>说明</h3><p>先宽召回、后分类筛选。没有 LLM 验证的条目以 heuristic 标注；所有标题、作者、链接都来自来源元数据而非 LLM 生成。</p></div>
{% if report.cross_insights %}<section><h2>💡 跨领域洞察</h2>{% for insight in report.cross_insights %}<div class="card"><h3>{{ insight.title }}</h3><p>{{ insight.content }}</p></div>{% endfor %}</section>{% endif %}
{% for track in report.paper_tracks %}
<section><h2>{{ track.track_name }}</h2><p class="muted">本周相关候选 {{ track.available }} 篇</p>
{% for item_type, label in [('recommended','🌟 精选'),('worth_knowing','📖 值得知道')] %}
{% set items = track[item_type] %}{% if items %}<h3>{{ label }}</h3>
{% for paper in items %}<article class="card"><h3><a href="{{ paper.url }}" rel="noopener noreferrer" target="_blank">{{ paper.title }}</a></h3>
<p class="meta">{{ paper.authors }} · {{ paper.published }} · 评分 {{ paper.total_score }} ({{ paper.score_origin }})
{% if track.track_id == 'rsi' %}<span class="tag">RSI {{ paper.rsi_level }}</span>{% endif %}</p>
<p>{{ paper.core_contribution or paper.abstract[:380] }}</p>
{% if item_type == 'recommended' %}<p><b>技术亮点：</b>{{ paper.technical_highlights or '摘要未说明' }}</p>
<p><b>应用方向：</b>{{ paper.application_direction or '需要阅读全文评估' }}</p>{% endif %}
{% if paper.soft_exclusion %}<p class="muted">⚠️ 与机器人/具身/世界模型关键词交叉，建议复核论文主要贡献</p>{% endif %}
</article>{% endfor %}{% endif %}{% endfor %}</section>{% endfor %}
<section><h2>🏆 本周重点阅读</h2><ol>{% for p in report.top_papers %}<li><a href="{{ p.url }}">{{ p.title }}</a> <span class="muted">{{ p.track }}</span></li>{% endfor %}</ol></section>
<details><summary>获取与筛选诊断</summary><pre>{{ metrics_json }}</pre></details>
</main></body></html>"""


def render_html(report: dict) -> str:
    env = Environment(loader=BaseLoader(), autoescape=select_autoescape(["html", "xml"]))
    return env.from_string(HTML_TEMPLATE).render(report=report, metrics_json=json.dumps(report["diagnostics"], ensure_ascii=False, indent=2))


def webhook_notify(c: Config, http: HttpClient, report: dict) -> None:
    if not c.webhook_url:
        return
    owner = os.getenv("GITHUB_REPOSITORY_OWNER", "")
    repo = os.getenv("GITHUB_REPOSITORY", "").split("/")[-1]
    page_url = f"https://{owner}.github.io/{repo}/" if owner and repo else "本地生成 reports/"
    payload = {"msgtype": "text", "text": {"content": f"🤖 {report['report_date']} AI 论文周报已更新\n{page_url}"}}
    try:
        http.request("POST", c.webhook_url, json_body=payload)
        LOG.info("Webhook sent")
    except Exception as exc:
        LOG.warning("Webhook failed: %s", exc)


def execute(c: Config, input_json: Path | None = None) -> dict:
    clock = now_utc()
    out = c.out_dir
    data_dir, report_dir = out / "data", out / "reports"
    data_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    http = HttpClient(c)
    fetcher = Fetcher(c, http, clock)
    if input_json:
        fresh = load_json(input_json, [])
        if not isinstance(fresh, list):
            raise ValueError("--input-json must be a JSON array of candidate dicts")
        LOG.info("Using offline candidate input: %s rows", len(fresh))
    else:
        fresh = fetcher.fetch_all()
    stats: dict[str, Any] = {"run_at": clock.isoformat(), "report_days": c.report_days,
                             "discovery_days": c.discovery_days, "raw_fetched": len(fresh),
                             "per_query": fetcher.per_query, "source_metrics": dict(fetcher.metrics)}
    fresh = merge_records(fresh)
    stats["fresh_deduplicated"] = len(fresh)
    history_path = data_dir / "paper_history.json"
    history = load_json(history_path, [])
    if not isinstance(history, list):
        history = []
    # Cache can fill transient source gaps; it NEVER makes an old paper appear new.
    retained = [p for p in history if recent(p.get("activity_date"), c.state_retention_days, clock)]
    combined = merge_records(retained + fresh)
    for p in combined:
        p.pop("assessment", None)  # do not reuse previous run's model scores
    stats["cache_retained"] = len(retained)
    stats["merged_with_cache"] = len(combined)
    save_json(history_path, combined)
    window = [p for p in combined if recent(p.get("activity_date"), c.discovery_days, clock)]
    stats["discovery_window_candidates"] = len(window)
    if not window:
        save_json(data_dir / "latest_diagnostics.json", stats)
        raise RuntimeError("No papers in discovery window. Inspect data/latest_diagnostics.json for API errors.")
    stats["soft_exclusion_flagged"] = sum(bool(p.get("soft_exclusion")) for p in window)
    stats["source_coverage"] = dict(Counter(src for p in window for src in p.get("sources", [])))
    llm = LLM(c, http)
    score_stats = score_candidates(window, c, clock, llm)
    stats["scoring"] = dict(score_stats)
    stats["classified_discovery_tracks"] = dict(Counter(p["assessment"]["track"] for p in window))
    stats["rsi_levels_discovery"] = dict(Counter(p["assessment"]["rsi_level"] for p in window
                                               if p["assessment"]["track"] == "rsi"))
    stats["rsi_source_coverage"] = dict(Counter(source for p in window
                                                   if p["assessment"]["track"] == "rsi"
                                                   for source in p.get("sources", [])))
    selections, top = select_for_report(window, c, clock)
    stats["report_candidates"] = sum(selections[t]["available"] for t in TRACKS)
    stats["report_tracks"] = {t: selections[t]["available"] for t in [*TRACKS, "framework"]}
    stats["report_top_count"] = len(top)
    stats["report_top_tracks"] = dict(Counter(p["assessment"]["track"] for p in top))
    summaries = summarize_papers(selections, top, llm, c)
    report = build_report(selections, top, summaries, clock, stats)
    enrich_editorial(report, selections, top, llm)
    stamp = report["report_date"]
    save_json(data_dir / "latest_candidates.json", window)
    save_json(data_dir / "latest_diagnostics.json", stats)
    save_json(report_dir / f"{stamp}.json", report)
    (report_dir / f"{stamp}.md").write_text(report_markdown(report), encoding="utf-8")
    page = render_html(report)
    (report_dir / f"{stamp}.html").write_text(page, encoding="utf-8")
    (out / "index.html").write_text(page, encoding="utf-8")
    LOG.info("Report: %s | candidates=%d, top=%d, RSI=%d", report_dir / f"{stamp}.md",
             stats["report_candidates"], len(top), selections["rsi"]["available"])
    for track, n in stats["report_tracks"].items():
        LOG.info("  %-15s %d", track, n)
    webhook_notify(c, http, report)
    return report


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Weekly AI/RSI paper discovery and reports")
    parser.add_argument("--discovery-days", type=int, default=21)
    parser.add_argument("--report-days", type=int, default=7)
    parser.add_argument("--top-count", type=int, default=12)
    parser.add_argument("--worth-per-track", type=int, default=10)
    parser.add_argument("--arxiv-max-pages", type=int, default=5)
    parser.add_argument("--max-llm-candidates", type=int, default=500)
    parser.add_argument("--out-dir", type=Path, default=Path("."))
    parser.add_argument("--input-json", type=Path, help="Use offline JSON array of normalized papers instead of fetching APIs")
    parser.add_argument("--no-llm", action="store_true", help="Skip all LLM requests; use transparent heuristic fallback")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = get_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    c = Config(discovery_days=args.discovery_days, report_days=args.report_days,
               top_count=args.top_count, worth_per_track=args.worth_per_track,
               arxiv_max_pages=args.arxiv_max_pages,
               max_llm_candidates=args.max_llm_candidates, out_dir=args.out_dir,
               use_llm=not args.no_llm,
               llm_api_key=os.getenv("LLM_API_KEY", ""),
               llm_base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1"),
               llm_model=os.getenv("LLM_MODEL", "deepseek-chat"),
               github_token=os.getenv("GITHUB_TOKEN", ""),
               hf_token=os.getenv("HF_TOKEN", ""), webhook_url=os.getenv("WEBHOOK_URL", ""))
    execute(c, args.input_json)


if __name__ == "__main__":
    main()
