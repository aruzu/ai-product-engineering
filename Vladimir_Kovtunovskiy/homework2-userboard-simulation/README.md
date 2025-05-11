# Spotify Review Insights 🔍

End‑to‑end pipeline that turns 60k+ Play‑Store reviews into **actionable product decisions** via multi‑agent simulation.

---

## 🚗 Quick start

```bash
# 1. Install deps
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Phase 1 – Preprocess and cluster reviews
python review_prep_pipeline.py --csv spotify_reviews.csv --out cluster_outputs

# 3. Phase 2 – Multi-agent user board simulation
python userboard_pipeline.py

# 4. Check artefacts ✨
open cluster_outputs/cluster_report.txt
open cluster_outputs/board_session_report.md
```

> **Note**
> – Set `OPENAI_API_KEY` in your environment (or a `.env`) for the multi-agent simulation.

---

## 🛠 Repo layout

```
.
├── spotify_reviews.csv           # raw CSV reviews
├── review_prep_pipeline.py       # preprocessing + clustering pipeline
├── userboard_pipeline.py         # multi-agent board simulation
├── requirements.txt              # project dependencies
├── cluster_outputs/             # generated artefacts
└── README.md
```

---

## 🧩 Phase 1 – Review Preprocessing & Clustering

The `review_prep_pipeline.py` script performs the following steps:

1. **Data Cleaning**
   - Filter non-English reviews
   - Normalize text (lowercase, ASCII conversion)
   - Standardize Spotify-related terms

2. **Clustering Pipeline**
   - Generate sentence embeddings
   - Apply dimensionality reduction
   - Perform clustering
   - Extract key topics per cluster

3. **Outputs**
   - `cluster_report.txt`: Human-readable cluster analysis
   - `clusters_data.json`: Machine-readable cluster data

---

## 🤖 Phase 2 – Multi-Agent User Board Simulation

The `userboard_pipeline.py` script implements:

1. **Persona Creation**
   - Generate user personas based on cluster insights
   - Define user characteristics and pain points

2. **Board Simulation**
   - Multi-agent discussion of product features
   - Structured debate format
   - Decision tracking and rationale

3. **Outputs**
   - `board_session_report.md`: Detailed session transcript
   - `board_session.log`: Structured execution log

---

## 📈 Extending the project

* **A/B Testing** – Integrate with analytics platforms to track feature impact
* **Dashboard** – Create interactive visualization of review clusters
* **Model Updates** – Experiment with different embedding models

---

## 👥 Author

Made with ❤️ by Vladimir Kovtunovskiy.