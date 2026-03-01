# Data curator Reference — Ghana LLM Datagen

Everything you need to curate this LLM data generation project.

---

## Repo Structure

```
ghana-llm-datagen/
├── run.py                          ← volunteers run this (do not edit after sharing)
├── requirements.txt
├── ultrachat_sample.csv            ← style reference, commit this to the repo
├── .env                            ← your API keys, NEVER commit this
├── scripts/
│   ├── DATA_CURATOR_README.md      ← you are here
│   ├──generate_codes.py            ← you run this to generate volunteer codes
│   ├──merge_results.py             ← merges all volunteer .xz submissions
│   └──.env                         ← contains your API keys, NEVER commit this
└── .github/
    └── ISSUE_TEMPLATE/
        └── result_submission.md    ← pre-fills the GitHub submission form
```

---

## One-Time Setup

### 1. Count your CSV rows
```bash
python -c "import pandas as pd; print(len(pd.read_csv('news_data.csv')))"
python -c "import pandas as pd; print(len(pd.read_csv('research_data.csv')))"
```

### 2. Fill in `.env`
```
NVIDIA_KEY_1=nvapi-...
NVIDIA_KEY_2=nvapi-...
NVIDIA_KEY_3=nvapi-...
NVIDIA_KEY_4=nvapi-...
NVIDIA_KEY_5=nvapi-...
```

### 3. Fill in `generate_codes.py`
Open `generate_codes.py` and set:
```python
NEWS_CSV_PATH     = "/path/to/news_data.csv"
RESEARCH_CSV_PATH = "/path/to/research_data.csv"
NUM_VOLUNTEERS    = 5
```

### 4. Update `run.py` config block
Open `run.py` and set these four lines near the top:
```python
GITHUB_REPO       = "yourusername/ghana-llm-datagen"
RELEASE_TAG       = "v1.0-data"
NEWS_FILENAME     = "news_data.csv"
RESEARCH_FILENAME = "research_data.csv"
```

### 5. Upload data files to GitHub Releases
- Go to your repo → **Releases** → **Create a new release**
- Tag: `v1.0-data`
- Title: `Dataset Files`
- Attach both `news_data.csv` and `research_data.csv`
- Publish

### 6. Commit everything to GitHub
```bash
git add run.py generate_codes.py requirements.txt ultrachat_sample.csv \
        scripts/ .github/ README.md .gitignore
git commit -m "Initial setup"
git push
```
Make sure `.env` and `volunteer_codes.json` are NOT committed — they are in `.gitignore`.

---

## Generating Volunteer Codes

```bash
python generate_codes.py
```

Prints a table of 5 codes and saves a backup to `volunteer_codes.json`:

```
#  TYPE       ROW RANGE              COUNT    CODE
1  news       0 – 2,400              2,400    eyJ0Ijo...
1  research   0 – 1,500              1,500    eyJ0Ijo...
...
```

Send **one code per volunteer**. Each code covers both their news slice and
research slice — they run the script once and it handles both automatically.

Keep `volunteer_codes.json` private — it contains your API keys.

---

## Tracking Volunteer Progress

Submissions come in as GitHub issues tagged **results**.

View all submissions:
```
https://github.com/YOUR_USERNAME/ghana-llm-datagen/issues?q=label%3Aresults
```

For each submission:
1. Download the two `.xz` attachments from the issue
2. Drop them into your local `results/` folder
3. Repeat for all volunteers before merging

---

## Merging Results

Once you have all `.xz` files in `results/`:

```bash
python scripts/merge_results.py
```

Default output is `final_dataset.jsonl`. To customise:

```bash
python scripts/merge_results.py \
    --results-dir ./results \
    --output ./final_dataset.jsonl
```

The merge script:
- Reads both `.xz` and `.jsonl` files — no manual decompression needed
- Deduplicates by `chunk_id` — safe to re-run if you add more files later
- Excludes records with `parse_error: true`
- Prints a per-file and overall summary

---

## If a Volunteer Has Problems

**They get an auth error (401/403):**
Their API key may be invalid or exhausted. Generate a new code with a
replacement key and resend. Their progress is saved so they resume from
where they left off.

**They need to restart from scratch:**
They can delete their local `results/` folder and re-run with the same code.

**They processed the wrong rows / wrong type:**
Their `chunk_id`s won't overlap with other volunteers so their output is safe
to merge anyway — the deduplication in `merge_results.py` handles any overlap.

**You need to add more volunteers later:**
Update `NUM_VOLUNTEERS` in `generate_codes.py` and re-run. Codes for
already-completed rows won't produce duplicate data in the final merge.

---

## Key Files — Quick Reference

| File | Who uses it | Purpose |
|---|---|---|
| `.env` | Data curator only | Stores API keys |
| `generate_codes.py` | Data curator only | Generates volunteer codes |
| `volunteer_codes.json` | Data curator only | Backup of all codes + keys |
| `run.py` | Volunteers | The only script volunteers touch |
| `ultrachat_sample.csv` | Auto-loaded by `run.py` | Style reference for prompts |
| `scripts/merge_results.py` | Data curator only | Combines all submissions |
| `results/*.xz` | Data curator only | Downloaded from GitHub issues |
