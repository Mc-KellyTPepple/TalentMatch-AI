"""
TalentMatch AI - live vacancy sources.

Pulls CURRENT job listings from two free, key-less public APIs:

  * Remotive   https://remotive.com/api/remote-jobs          (remote jobs)
  * Arbeitnow  https://www.arbeitnow.com/api/job-board-api   (Europe + remote,
                                                              visa-sponsorship flag)

Rules respected
---------------
* Remotive asks consumers to cache and to call at most ~4 times per day, to
  show the source name and to link back to the listing URL. This module
  therefore refreshes at most once per LIVE_JOBS_TTL_HOURS (default 6) and
  every job carries `source` + `url`.
* Listings are held in memory only (trimmed text), never written to disk.
* A failing source never breaks the app: stale cache is served and the
  problem is reported through `status()`.
"""

from __future__ import annotations

import html
import json
import os
import re
import threading
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

USER_AGENT = "TalentMatch-AI/2.0 (+career-matching tool; contact via repository owner)"
REMOTIVE_URL = "https://remotive.com/api/remote-jobs"
ARBEITNOW_URL = "https://www.arbeitnow.com/api/job-board-api"

HTTP_TIMEOUT = 8
MAX_AGE_DAYS = 75
MAX_TEXT_CHARS = 3500
ARBEITNOW_PAGES = 3

_TAG_RE = re.compile(r"<[^>]+>")
_BR_RE = re.compile(r"(?i)<\s*(?:br|/p|/li|/h\d|/div)\s*/?>")
_WS_RE = re.compile(r"[ \t\r\f\v]+")


def strip_html(raw: Any, limit: int = MAX_TEXT_CHARS) -> str:
    text = str(raw or "")
    text = _BR_RE.sub("\n", text)
    text = _TAG_RE.sub(" ", text)
    text = html.unescape(text)
    text = _WS_RE.sub(" ", text)
    text = re.sub(r"\n\s*\n+", "\n", text).strip()
    return text[:limit]


def _http_json(url: str) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT) as response:
        return json.loads(response.read().decode("utf-8", errors="replace"))


def _parse_date(value: Any) -> Optional[datetime]:
    if value in (None, ""):
        return None
    try:
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        text = str(value).replace("Z", "+00:00")
        parsed = datetime.fromisoformat(text)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Source adapters (pure functions: payload -> list of normalised jobs)
# ---------------------------------------------------------------------------

def normalise_remotive(payload: Any) -> List[Dict[str, Any]]:
    jobs = payload.get("jobs", []) if isinstance(payload, dict) else []
    result = []
    for item in jobs:
        if not isinstance(item, dict) or not item.get("url") or not item.get("title"):
            continue
        posted = _parse_date(item.get("publication_date"))
        result.append({
            "id": f"remotive-{item.get('id')}",
            "source": "Remotive",
            "title": str(item.get("title")).strip(),
            "company": str(item.get("company_name") or "").strip(),
            "location": str(item.get("candidate_required_location") or "Worldwide").strip(),
            "remote": True,
            "visa": False,
            "url": str(item["url"]),
            "posted": posted.date().isoformat() if posted else "",
            "_posted_dt": posted,
            "job_type": str(item.get("job_type") or "").replace("_", " "),
            "salary": str(item.get("salary") or "").strip(),
            "category": str(item.get("category") or "").strip(),
            "tags": [str(t) for t in (item.get("tags") or [])][:10],
            "text": strip_html(item.get("description")),
        })
    return result


def normalise_arbeitnow(payload: Any, visa_slugs: Optional[set] = None) -> List[Dict[str, Any]]:
    jobs = payload.get("data", []) if isinstance(payload, dict) else []
    visa_slugs = visa_slugs or set()
    result = []
    for item in jobs:
        if not isinstance(item, dict) or not item.get("url") or not item.get("title"):
            continue
        posted = _parse_date(item.get("created_at"))
        slug = str(item.get("slug") or item.get("url"))
        result.append({
            "id": f"arbeitnow-{slug}",
            "source": "Arbeitnow",
            "title": str(item.get("title")).strip(),
            "company": str(item.get("company_name") or "").strip(),
            "location": str(item.get("location") or "").strip(),
            "remote": bool(item.get("remote")),
            "visa": slug in visa_slugs,
            "url": str(item["url"]),
            "posted": posted.date().isoformat() if posted else "",
            "_posted_dt": posted,
            "job_type": ", ".join(str(t) for t in (item.get("job_types") or [])),
            "salary": "",
            "category": ", ".join(str(t) for t in (item.get("tags") or [])[:3]),
            "tags": [str(t) for t in (item.get("tags") or [])][:10],
            "text": strip_html(item.get("description")),
        })
    return result


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

class LiveJobStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: List[Dict[str, Any]] = []
        self._fetched_at = 0.0
        self._refreshing = False
        self._version = 0
        self._status: Dict[str, Any] = {"remotive": "not fetched", "arbeitnow": "not fetched"}

    # -- configuration ------------------------------------------------------
    @staticmethod
    def enabled() -> bool:
        return os.getenv("LIVE_JOBS", "1").strip().lower() not in {"0", "false", "no", "off"}

    @staticmethod
    def ttl_seconds() -> float:
        try:
            return max(0.5, float(os.getenv("LIVE_JOBS_TTL_HOURS", "6"))) * 3600
        except ValueError:
            return 6 * 3600

    # -- state ----------------------------------------------------------------
    @property
    def version(self) -> int:
        return self._version

    def status(self) -> Dict[str, Any]:
        age = int(time.time() - self._fetched_at) if self._fetched_at else None
        return {
            "enabled": self.enabled(),
            "jobs_cached": len(self._jobs),
            "cache_age_seconds": age,
            "refreshing": self._refreshing,
            "sources": dict(self._status),
        }

    def is_stale(self) -> bool:
        return (not self._jobs) or (time.time() - self._fetched_at) > self.ttl_seconds()

    # -- refresh --------------------------------------------------------------
    def refresh(self) -> int:
        """Fetch both sources. Keeps the previous cache if everything fails."""
        if not self.enabled():
            return 0
        with self._lock:
            if self._refreshing:
                return len(self._jobs)
            self._refreshing = True
        try:
            collected: List[Dict[str, Any]] = []

            try:
                collected += normalise_remotive(_http_json(REMOTIVE_URL))
                self._status["remotive"] = "ok"
            except Exception as exc:  # noqa: BLE001
                self._status["remotive"] = f"error: {type(exc).__name__}"

            try:
                visa_slugs: set = set()
                try:
                    visa_payload = _http_json(ARBEITNOW_URL + "?visa_sponsorship=true")
                    visa_slugs = {str(j.get("slug")) for j in visa_payload.get("data", []) if isinstance(j, dict)}
                except Exception:  # noqa: BLE001 - the visa flag is optional
                    pass
                for page in range(1, ARBEITNOW_PAGES + 1):
                    payload = _http_json(f"{ARBEITNOW_URL}?page={page}")
                    batch = normalise_arbeitnow(payload, visa_slugs)
                    if not batch:
                        break
                    collected += batch
                self._status["arbeitnow"] = "ok"
            except Exception as exc:  # noqa: BLE001
                self._status["arbeitnow"] = f"error: {type(exc).__name__}"

            cutoff = datetime.now(timezone.utc) - timedelta(days=MAX_AGE_DAYS)
            fresh, seen = [], set()
            for job in collected:
                posted = job.get("_posted_dt")
                if posted and posted < cutoff:
                    continue
                key = job["url"]
                if key in seen:
                    continue
                seen.add(key)
                job.pop("_posted_dt", None)
                fresh.append(job)

            if fresh:
                with self._lock:
                    self._jobs = fresh
                    self._fetched_at = time.time()
                    self._version += 1
            return len(self._jobs)
        finally:
            self._refreshing = False

    def refresh_async(self) -> None:
        if not self.enabled() or self._refreshing:
            return
        threading.Thread(target=self.refresh, name="live-jobs-refresh", daemon=True).start()

    def get_jobs(self, wait_if_empty: bool = True) -> List[Dict[str, Any]]:
        """Return cached jobs, refreshing in the background when stale."""
        if not self.enabled():
            return []
        if not self._jobs and wait_if_empty:
            self.refresh()
        elif self.is_stale():
            self.refresh_async()
        return self._jobs


store = LiveJobStore()
