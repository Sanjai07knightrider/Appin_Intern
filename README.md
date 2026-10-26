# EduPredict AI — Intelligent Student Performance Prediction Platform

**EduPredict AI** is a production-grade, end-to-end Machine Learning web application designed to help academic advisors, educators, and department heads proactively forecast student academic trajectories, identify at-risk learners early, and deliver targeted interventions.

Built with **Streamlit**, **Scikit-learn**, **Pandas**, and **Plotly**, EduPredict AI implements real, uncompromised machine learning workflows with honest holdout evaluation, interactive cohort analytics, and dual-format data export.

---

## 🌟 Key Features

1. **Excel Upload & Strict Data Validation**
   - Supports `.xlsx` student academic workbooks.
   - Built-in schema verification, numeric range checks (`0–100%`, `0–24 hrs`), duplicate student ID detection, and category normalization.
   - Automatically separates historical labelled cohorts from unlabelled students requiring prediction.
   - Provides a downloadable, pre-formatted Excel template.

2. **Real Machine Learning Pipeline (Zero Hardcoded Metrics)**
   - Preprocessing with median imputation (`SimpleImputer`) and feature standardization (`StandardScaler`).
   - Stratified holdout test split (80% train / 20% test).
   - Baseline classifier (**Logistic Regression**) compared against an ensemble model (**Random Forest Classifier**).
   - Holdout evaluation reporting **Accuracy**, **Macro Precision**, **Macro Recall**, **Macro F1-Score**, per-class **Classification Reports**, and an annotated **Confusion Matrix**.
   - Final retraining on all historical records before inferring unlabelled students with probability confidence scores.

3. **Premium Emerald Dashboard (3 Views)**
   - **View 1 — Overview**: High-level KPIs, model holdout performance comparison, interactive category distribution donut and bar charts, and recent prediction previews.
   - **View 2 — Student Explorer**: Real-time student search, category and record-type filters, sortable results table, individual deep-dive student scorecards with risk diagnosis, and instant export to **Excel** and **CSV**.
   - **View 3 — Performance Insights**: Interactive Plotly scatter plots (Study vs. Previous Marks, Sleep vs. Attendance), feature distribution boxplots, relative ML feature importance weights, and full correlation matrix heatmaps.

---

## 🏗️ Project Architecture

```
EduPredict AI/
├── .streamlit/
│   └── config.toml             # Custom emerald styling & server configuration
├── data/
│   └── sample_students.xlsx    # Pre-generated realistic synthetic demonstration dataset
├── src/
│   ├── __init__.py
│   ├── constants.py            # Feature names, ranges, color codes, category labels
│   ├── data_processor.py       # Excel validation, schema checks, template generator & exporters
│   ├── ml_pipeline.py          # Scikit-learn pipelines, holdout evaluation, confidence scoring
│   ├── synthetic_data.py       # Realistic synthetic student academic dataset generator
│   └── ui_components.py        # Premium CSS styling, KPI metric cards & Plotly chart builders
├── tests/
│   ├── test_export.py          # Automated tests for Excel/CSV exports & templates
│   ├── test_pipeline.py        # Automated tests for ML training, holdout evaluation & predictions
│   └── test_validation.py      # Automated tests for schema, duplicates, ranges & categories
├── app.py                      # Main Streamlit web application
├── requirements.txt            # Production dependencies
├── .gitignore                  # Standard Python & Streamlit ignore rules
└── README.md                   # Complete documentation
```

---

## 📋 Data Schema Requirements

Uploaded workbooks must contain an active sheet with the following 8 columns:

| Column | Type | Allowed Range | Description |
|---|---|---|---|
| `student_id` | Text | Unique string | Unique identifier for each student (e.g. `STU-1001`) |
| `attendance_pct` | Float | `0.0 – 100.0` | Overall attendance percentage |
| `previous_marks_pct` | Float | `0.0 – 100.0` | Previous term / semester academic marks |
| `assignment_avg_pct` | Float | `0.0 – 100.0` | Continuous internal assignment average |
| `internal_marks_pct` | Float | `0.0 – 100.0` | Internal examination score percentage |
| `study_hours_per_day` | Float | `0.0 – 24.0` | Estimated daily self-study hours |
| `sleep_hours_per_day` | Float | `0.0 – 24.0` | Average daily sleep duration in hours |
| `performance_category` | Text | `At Risk`, `Average`, `Good` *(or blank)* | Historical classification label. Leave blank or empty for rows requiring prediction. |

---

## 🚀 Local Installation & Running (Windows)

### Prerequisites
- Python 3.10+ (tested on Python 3.10, 3.11, 3.12, 3.14)
- Git (optional, for repository management)

### Step 1: Clone or Navigate to Project
```powershell
cd "d:\intern\Appin Technology"
```

### Step 2: Set Up Virtual Environment (Recommended)
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 4: Run Automated Tests
Verify all 21 automated tests pass:
```powershell
python -m pytest -v
```

### Step 5: Launch EduPredict AI
```powershell
streamlit run app.py
```
Your default web browser will automatically open:
`http://localhost:8501`

---

## 🧪 Running the Demonstration Workflow

1. **Launch the application** — Upon opening, the app automatically initializes with a realistic demonstration cohort (90 historical labelled students + 20 unlabelled students).
2. **Review View 1 (Overview)** — Inspect the live holdout evaluation metrics, the comparison between Logistic Regression and Random Forest, and the category distribution donut chart.
3. **Explore View 2 (Student Explorer)** — Search for any student ID (e.g., `STU-1005`), filter by predicted category (`At Risk`, `Average`, `Good`), inspect individual risk indicators, and download results in Excel or CSV format.
4. **Inspect View 3 (Performance Insights)** — Analyze empirical correlations between study habits, sleep, attendance, and exam marks.
5. **Upload Custom Excel Files** — Switch to **"Upload Custom Excel (.xlsx)"** in the sidebar or download the provided blank template to test your own institution's data.

---

## ☁️ Deploying to Streamlit Community Cloud

Deploying EduPredict AI is free and takes under 3 minutes:

### Step 1: Push Project to GitHub
1. Create a new GitHub repository named `edupredict-ai`.
2. In your local terminal, initialize Git, commit, and push:
```bash
git init
git add .
git commit -m "Initial commit: Complete EduPredict AI application"
git branch -M main
git remote add origin https://github.com/<your-username>/edupredict-ai.git
git push -u origin main
```

### Step 2: Deploy on Streamlit Cloud
1. Visit [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
2. Click **"Create app"** or **"New app"**.
3. Select your repository: `<your-username>/edupredict-ai`.
4. Set the **Main file path** to: `app.py`.
5. Click **"Deploy!"**.
6. Streamlit Community Cloud will automatically install dependencies from `requirements.txt` and launch your live public web app with a permanent URL.

---

## ⚖️ Academic Integrity & Ethical Disclaimers

1. **Decision Support, Not Definitive Labels**: Model predictions are probabilistic estimates derived from historical numerical patterns. They are intended solely to assist educators and academic counselors in allocating mentoring resources. They should **never** be used as automated punitive measures or definitive judgments about a student's potential.
2. **Correlation vs. Causation**: Feature relationships identified in the insights view represent empirical statistical associations within the uploaded dataset. High attendance or study hours correlate with higher marks, but individual student outcomes are influenced by qualitative factors (health, socioeconomic context, pedagogy) not captured in numerical feature vectors.
3. **Synthetic Demonstration Data**: The default sample dataset was synthetically generated using normal distributions with realistic academic noise to showcase the complete pipeline safely without exposing confidential student records. Metrics measured on synthetic cohorts serve as illustrative demonstrations of pipeline functionality rather than empirical pedagogical benchmarks.

---

## 📄 License
MIT License. Open-source for academic, educational, and institutional research purposes.
