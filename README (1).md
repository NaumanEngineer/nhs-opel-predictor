# 🏥 NHS ICB OPEL Level Predictor
### AI-Powered Operational Pressure Forecasting for NHS Integrated Care Boards

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange?style=flat-square&logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?style=flat-square&logo=streamlit)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-green?style=flat-square)
![NHS](https://img.shields.io/badge/NHS-Governance%20Ready-005EB8?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

---

## 📋 Project Overview

A complete end-to-end machine learning project designed for NHS Integrated Care Board (ICB) operational planning teams. The system predicts daily **OPEL (Operational Pressure Escalation Level)** classifications — from OPEL 1 (normal operations) to OPEL 4 (maximum escalation) — using 19 operational, workforce, and environmental features.

Built with NHS clinical governance requirements at the core: interpretable model, SHAP explainability, formal governance documentation, and a human-in-the-loop interactive dashboard.

> ⚠️ **Important:** This project uses synthetic data and is intended for portfolio demonstration and decision-support research only. Not for clinical deployment without full DCB0129/DCB0160 compliance and DPIA.

---

## 🎯 Business Problem

NHS Trusts and ICBs face daily operational pressure decisions that affect patient safety, staff wellbeing, and resource allocation. OPEL escalation levels determine when additional resources are deployed, when elective procedures are cancelled, and when system-wide alerts are triggered.

**The challenge:** Operational teams currently rely on manual assessment of dozens of metrics. This project provides a data-driven decision-support layer that:
- Predicts OPEL level from daily operational data
- Explains *why* each prediction was made (essential for NHS governance)
- Presents findings in a clinician-friendly dashboard

---

## 📊 Key Results

| Metric | Value |
|--------|-------|
| Overall Accuracy | **92.0%** |
| OPEL 4 Recall (safety-critical) | **90.0%** |
| ROC-AUC (all classes) | **0.988 – 0.997** |
| Training observations | **2,190** daily records |
| Trusts represented | **3** (2 Acute, 1 Community) |
| Features used | **19** (operational, workforce, environmental) |

---

## 🗂️ Project Structure

```
nhs-opel-predictor/
│
├── 📊 data/
│   ├── nhs_icb_synthetic_raw.csv          # Raw synthetic dataset (2,190 rows)
│   └── nhs_icb_clean.csv                  # Cleaned dataset with engineered KPIs
│
├── 🤖 model/
│   ├── nhs_opel_baseline_model.pkl        # Trained logistic regression model
│   └── nhs_opel_scaler.pkl                # Feature scaler (StandardScaler)
│
├── 📓 notebooks/
│   └── nhs_opel_model.ipynb               # Full Colab notebook (Sessions 3-4)
│
├── 📈 outputs/
│   ├── feature_importance.png             # Feature importance chart
│   ├── shap_summary_opel4.png             # SHAP summary plot (OPEL 4)
│   ├── shap_waterfall.png                 # SHAP waterfall (single prediction)
│   ├── shap_dependence.png                # SHAP dependence plot
│   └── roc_curve.png                      # ROC curves (all classes)
│
├── 📄 docs/
│   ├── NHS_OPEL_Governance_Report.docx    # Full clinical governance report
│   └── NHS_Clinical_Governance_Review.docx # Independent governance review
│
├── 🖥️ app.py                              # Streamlit dashboard
├── 📋 SETUP.md                            # Installation guide
└── 📖 README.md                           # This file
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/nhs-opel-predictor.git
cd nhs-opel-predictor

# Install dependencies
pip install streamlit scikit-learn joblib pandas numpy matplotlib shap

# Run the dashboard
streamlit run app.py
```

Dashboard opens at `http://localhost:8501`

---

## 🏗️ Project Methodology

### Session 1 — Problem Definition & Data Generation
- Defined clinical problem: predict OPEL escalation for ICB planning
- Designed synthetic NHS dataset with realistic causal relationships
- 2,190 daily observations across 3 trusts (WGH, NRI, SCH)
- Embedded deliberate data quality issues for cleaning practice

### Session 2 — Data Cleaning & KPI Engineering *(Excel)*
- Identified and corrected 8 planted data errors (impossible values)
- Trust-specific median imputation for invalid readings
- Forward-fill within trust for missing values (time-series aware)
- Engineered 5 clinically meaningful KPIs:

| KPI | Formula | Clinical rationale |
|-----|---------|-------------------|
| `ae_pressure_index` | Breach rate × 0.4 + Wait × 0.3 + NCTR × 0.3 | Combined A&E stress signal |
| `bed_social_care_ratio` | Social care delay ÷ Bed occupancy | Social care blockage indicator |
| `workforce_stress_index` | Unfilled shifts × 0.5 + Sickness × 0.5 | Staff capacity index |
| `system_flow_score` | Occupancy × 0.4 + Social care × 0.3 + GP lag × 0.3 | End-to-end pathway flow |
| `rag_status` | IF pressure ≥ 0.7 → Red, ≥ 0.4 → Amber, else Green | Operational RAG flag |

### Session 3 — Predictive Model

**Model selection rationale:** Logistic Regression was chosen over more complex alternatives (Random Forest, XGBoost) because:
- Inherently interpretable — essential for NHS clinical governance
- Coefficients are clinically explainable
- Suitable for the feature set size and class structure
- Aligns with NHS AI Lab guidance on regulated clinical AI

**Baseline vs Tuned model decision:**

| Model | Accuracy | OPEL 4 Precision | OPEL 4 Recall |
|-------|----------|-----------------|---------------|
| Baseline (C=1) | 92.0% | **0.95** | **0.90** |
| Tuned (C=100) | 93.2% | 0.93 | 0.88 |
| **Winner** | Tuned | **Baseline** | **Baseline** |

> **Clinical governance decision:** Baseline model selected for deployment. Despite lower overall accuracy, it outperforms on OPEL 4 recall — the safety-critical metric. Missing a real OPEL 4 day carries greater patient risk than a false alarm. This decision is documented in the governance report.

### Session 4 — SHAP Explainability

Applied SHAP (SHapley Additive exPlanations) for both global and individual prediction explanation:

**Key findings:**
- `bed_occupancy_pct` — dominant predictor (coefficient 3.9×, far exceeding all others)
- `ae_4hr_breach_pct` — second strongest predictor
- `temperature_c` — third strongest, with **reversed directionality** (low temperature → OPEL 4 risk)
- Bed occupancy and temperature are **correlated** — cold days are also high-occupancy days (compound winter effect)
- Three engineered KPIs ranked in top 7 features — validating KPI design

### Session 5 — Interactive Dashboard
- Streamlit dashboard for live OPEL prediction
- 19 operational input sliders with clinical descriptions
- Real-time prediction with confidence scores across all 4 classes
- Key drivers displayed with values and direction
- NHS branding with embedded safety and governance notices

---

## 📈 Model Performance

### ROC Curves
All four OPEL classes achieve AUC > 0.988, indicating near-perfect discrimination.

| Class | AUC |
|-------|-----|
| OPEL 1 | 0.997 |
| OPEL 2 | 0.990 |
| OPEL 3 | 0.988 |
| OPEL 4 | 0.994 |

### Confusion Matrix (Baseline Model)
```
              Predicted
              1    2    3    4
Actual  1  [ 42    5    0    0 ]
        2  [  3  179    7    0 ]
        3  [  0    8  113    4 ]
        4  [  0    0    8   69 ]
```
No OPEL 4 day was ever predicted as OPEL 1 or 2 — critical for patient safety.

---

## 🔍 SHAP Explainability Findings

### Global Feature Importance (OPEL 4)
Top predictors driving critical escalation risk:
1. `bed_occupancy_pct` — dominant signal (3.9 mean absolute coefficient)
2. `ae_4hr_breach_pct` — second strongest (2.2)
3. `temperature_c` — winter pressure signal (1.0)
4. `ae_pressure_index` — engineered KPI (1.0)
5. `red_flag_incidents` — patient safety events (1.0)

### Key Clinical Insight
*"The SHAP dependence plot revealed that bed occupancy and temperature are correlated risk factors — cold days are also high occupancy days. Winter pressure is a compounding effect, not additive. NHS winter planning must treat bed capacity and seasonal demand as a single integrated problem."*

---

## 📋 NHS Governance

This project follows NHS AI governance principles:

- ✅ **Interpretable model** — logistic regression, not black box
- ✅ **SHAP explainability** — global and individual prediction explanation
- ✅ **Human-in-the-loop** — dashboard designed as decision support, not autonomous trigger
- ✅ **Formal governance documentation** — model card, limitations, deployment conditions
- ✅ **Clinical safety case** — DCB0129/DCB0160 pathway documented
- ✅ **Synthetic data** — no patient-identifiable information

**Governance verdict:** Approved for pilot evaluation. Not approved for autonomous operational deployment without DCB0129 sign-off.

See `docs/NHS_OPEL_Governance_Report.docx` for full clinical safety documentation.

---

## 🛠️ Technology Stack

| Tool | Purpose |
|------|---------|
| Python 3.8+ | Core language |
| pandas / numpy | Data manipulation |
| scikit-learn | Model training and evaluation |
| SHAP | Explainability analysis |
| matplotlib | Visualisations |
| Streamlit | Interactive dashboard |
| joblib | Model serialisation |
| Microsoft Excel | Data cleaning and KPI engineering |
| Google Colab | Model development environment |

---

## 📄 Documentation

| Document | Description |
|----------|-------------|
| `NHS_OPEL_Governance_Report.docx` | Full clinical safety & AI governance report (9 sections, 240 paragraphs) |
| `NHS_Clinical_Governance_Review.docx` | Independent governance review with deployment recommendation |
| `SETUP.md` | Local installation and usage guide |

---

## 🎓 Learning Outcomes

This project was completed as part of an 18-month NHS Health & Care AI Roadmap. Skills demonstrated:

- ✅ NHS domain knowledge (OPEL, NCTR, ICB structure, winter pressure)
- ✅ End-to-end ML pipeline (data cleaning → model → evaluation → deployment)
- ✅ Clinical governance thinking (model selection based on recall, not just accuracy)
- ✅ Explainable AI (SHAP global and individual explanations)
- ✅ Production thinking (dashboard, setup instructions, model serialisation)
- ✅ Professional documentation (NHS-branded Word documents)
- ✅ Feature engineering with clinical justification

---

## ⚠️ Limitations & Disclaimer

- Model trained on **synthetic data** — real-world performance will differ
- **No temporal validation** — random train/test split used (chronological split recommended for production)
- **Three trusts only** — generalisation to wider ICB not demonstrated
- **Class imbalance** not addressed (OPEL 1: 10.2% vs OPEL 2: 38.5%)
- This is a **decision support tool** — not for autonomous clinical triggering
- Full list of limitations in governance report Section 4

---

## 📞 Contact

**Author:** Nauman Ismail Khan
**LinkedIn:** https://www.linkedin.com/in/nauman-khan-844969265/
**Email:**naumanismailkhan902@gmail.com

---

## 📜 License

This project is licensed under the MIT License. See `LICENSE` for details.

---

*Built as part of the NHS Health & Care AI 18-Month Roadmap — Phase 3: Machine Learning*
