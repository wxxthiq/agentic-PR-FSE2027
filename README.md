# Replication Package for empirical submission to **ACM FSE 2027**.

This repository provides all source code, deterministic taxonomies, exact matching pipelines, and Jupyter notebooks to reproduce all findings across Research Questions 1–3.


## 1. Repository Structure

.
├── README.md                  # Replication instructions
├── requirements.txt           # Pinned Python dependencies
├── src/
│   ├── __init__.py            # Python package initialization
│   └── pipeline_utils.py      # Core classification, matching, and caching utilities
├── notebooks/
│   ├── rq1.ipynb   # RQ1: Prevalence, chore analysis, and attrition audits
│   ├── rq2.ipynb      # RQ2: 1:1 matching, AFT survival models, and latency
│   └── rq3.ipynb  # RQ3: Bot filtering, McNemar tests, and governance logit
└── data/
    └── experimental/          # Local Parquet cache (auto-populated on first run)


---

## 2. Prerequisites & Quickstart

* **Python:** Python 3.10+
* **Storage:** ~5 GB free space (for cached Parquet datasets)

```bash
# 1. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate       # On Windows: .\venv\Scripts\activate

# 2. Install pinned dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Data Ingestion & Caching

Datasets (`hao-li/AIDev-7.6M` and `AISE-TUDelft/MOSAIC-agentic-3m`) are **streamed and cached automatically** from Hugging Face on the first notebook execution via `src/pipeline_utils.py`. No manual data downloading is required.

---

## 4. Execution & Artifact Mapping

Launch Jupyter:
```bash
jupyter lab
```

Execute the notebooks sequentially in `notebooks/`:

| Notebook | Focus | Paper Outputs Generated |
| :--- | :--- | :--- |
| `rq1.ipynb` | **RQ1: Prevalence & Chores** | Table 1 (Cohort Funnel), Figure 1 Data (Scope & Subdomains), Table 2 (Prevalence), Section 3.1 (Chores) |
| `rq2.ipynb` | **RQ2: Integration Latency** | Figure 2 Data (ECDF Distributions), Table 3 (AFT Models), Section 3.2 (ATT & Subdomain Bottlenecks) |
| `rq3.ipynb` | **RQ3: Review & Gatekeeping** | Table 4 (McNemar Tests), Section 3.3 (Stratum A Logit, Bimodal Latency Dynamics) |

---

## 5. Anonymity & Licensing

* **Double-Anonymous Review:** This package contains zero author names, emails, institutional affiliations, or personal repository paths.
* **License:** Source code is distributed under the **MIT License**; curated data definitions under **CC-BY 4.0**.
* **Archival:** Upon formal acceptance, this package will be permanently archived on Zenodo with a persistent Digital Object Identifier (DOI).
```
