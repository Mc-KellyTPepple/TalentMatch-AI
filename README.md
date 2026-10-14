# TalentMatch AI (v2)

Upload a CV and get:

- **Your profile** - skills found, estimated experience and level, education.
- **Career roles that fit you** - ranked, each with the skills you have, the core skills you are missing, where the role can lead, and one-click search links (LinkedIn, Indeed, Google Jobs).
- **Live job openings** - current vacancies from Remotive and Arbeitnow, scored against your CV, with eligibility for your country, seniority checks and a direct link to apply.
- **Skills to learn next**, **CV quick wins** and **interview practice** for your best-fit roles.

## How matching works

1. `matcher.py` extracts skills (clean vocabulary from `roles_kb.py`), estimates experience and level, and scores 59 career roles by weighted skill coverage + job-title evidence + TF-IDF similarity.
2. `live_jobs.py` caches current vacancies (refreshed at most every 6 hours, in memory only).
3. `ranking_engine.py` scores each vacancy: role alignment, skill overlap, wording similarity, recency, seniority fit and location eligibility. Weak matches are dropped rather than padded.
4. If `GROQ_API_KEY` is set, the top 10 are re-checked by an LLM and blended 60/40 with the local score. If Groq is unavailable the app falls back to the local score and says so.

No PyTorch / transformer model is loaded: the whole service uses ~130 MB RAM, starts in about a second, and fits Render's free tier.

## Deploy on Render

1. Push this folder to GitHub and create a **Web Service** (or use `render.yaml`).
2. Build command `pip install -r requirements.txt`; start command `uvicorn app:app --host 0.0.0.0 --port $PORT --workers 1`; health check path `/health`.
3. Environment variables (all optional): `GROQ_API_KEY` (server-side only), `GROQ_MODEL`, `LIVE_JOBS` (`0` disables live feeds), `LIVE_JOBS_TTL_HOURS`, `ANALYSES_PER_10_MIN`.
4. Open `/status` after deploying to confirm roles loaded, AI configured and feed status.

The free tier sleeps after ~15 minutes idle; the first request afterwards can take up to a minute.
If you host the web page somewhere else (e.g. Vercel), add `<script>window.TALENTMATCH_API_BASE = "https://your-app.onrender.com";</script>` before `app.js`.

## Run locally / test

```bash
pip install -r requirements-dev.txt
uvicorn app:app --reload
python -m pytest -q
```

## Customising

- **Add or tune roles**: edit `roles_kb.py` (core skills, secondary skills, title aliases, next roles). No retraining or artifact rebuild.
- **Tune strictness**: constants at the top of `ranking_engine.py` (`MIN_LOCAL_JOB_SCORE`, `MIN_GROQ_SCORE`, blend weights) and `STRONG_FIT` in `matcher.py`.
- **More job sources**: add a `normalise_*` adapter and fetch call in `live_jobs.py`.

## Data sources and terms

Live listings come from [Remotive](https://remotive.com) and [Arbeitnow](https://www.arbeitnow.com). Remotive requires attribution and a link back to each listing (both are shown in the UI) and asks for no more than ~4 API calls a day; the cache keeps this app well inside that. Do not republish Remotive listings on third-party job boards. CVs are processed in memory and never stored or logged.

## Limits worth knowing

- Skill extraction recognises the skills in `roles_kb.py`; unusual niche skills are not counted.
- Experience is estimated from stated years and date ranges.
- Scanned (image-only) PDFs have no text to read; the app asks for a text PDF or DOCX.
- Scores are compatibility estimates, not hiring predictions.
