TalentMatch AI matching improvement pack

1. Replace your existing ranking_engine.py with the supplied ranking_engine.py.
2. Run rebuild_job_index_clean.py ONCE from the TalentMatch project root.
3. Restart/redeploy the application.

Why both are needed:
- ranking_engine.py prevents weak matches and duplicate recommendations at request time.
- rebuild_job_index_clean.py removes duplicate source postings and preserves title/company
  metadata while regenerating aligned embeddings and TF-IDF artifacts.

Existing config.py paths are reused; no dataset path is hard-coded.

Expected behavior for a weakly related resume:
- weak HR/Data Scientist postings should not be shown merely because semantic similarity
  found generic words such as leadership, performance, team, or management.
- duplicate/near-identical postings should collapse to one.
- if there are no sufficiently relevant jobs, the API returns zero recommendations and
  the summary says "No Suitable Match" instead of presenting a 20–30% match as a recommendation.

Important:
The score is a compatibility estimate, not a probability of employment.
