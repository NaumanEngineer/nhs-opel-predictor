# NHS OPEL Level Predictor — Local Dashboard Setup

## Quick Start (3 minutes)

### Prerequisites
- Python 3.8+ installed on your computer
- Your model files: `nhs_opel_baseline_model.pkl` and `nhs_opel_scaler.pkl`

### Installation

1. **Create a project folder:**
```bash
mkdir nhs_opel_dashboard
cd nhs_opel_dashboard
```

2. **Create a Python virtual environment:**
```bash
python -m venv venv
source venv/bin/activate          # On Mac/Linux
# or
venv\Scripts\activate             # On Windows
```

3. **Install dependencies:**
```bash
pip install streamlit scikit-learn joblib pandas numpy matplotlib
```

4. **Copy files into the folder:**
   - `app.py` (the dashboard code — you have this)
   - `nhs_opel_baseline_model.pkl` (your trained model)
   - `nhs_opel_scaler.pkl` (your scaler)

5. **Run the dashboard:**
```bash
streamlit run app.py
```

Your browser will open automatically to `http://localhost:8501`

---

## What the Dashboard Does

1. **Input operational metrics** — sliders for bed occupancy, A&E performance, workforce data
2. **Get OPEL prediction** — instant classification into OPEL 1, 2, 3, or 4
3. **See confidence scores** — how confident is the model on each level?
4. **Identify key drivers** — which metrics most influenced this prediction?
5. **Safety warnings** — governance reminders built in

---

## How to Use It

1. Adjust the sliders to match today's operational situation
2. The prediction updates in real-time
3. Check which metrics are driving the result
4. Use this alongside clinical judgement for operational planning
5. This is a **decision support tool**, not an autonomous trigger

---

## Troubleshooting

**"ModuleNotFoundError: No module named 'joblib'"**
→ Run: `pip install joblib`

**"Model files not found"**
→ Make sure `nhs_opel_baseline_model.pkl` and `nhs_opel_scaler.pkl` are in the same folder as `app.py`

**Port 8501 already in use**
→ Run: `streamlit run app.py --server.port 8502`

---

## File Structure

```
nhs_opel_dashboard/
├── app.py
├── nhs_opel_baseline_model.pkl
├── nhs_opel_scaler.pkl
└── README.md (this file)
```

---

## Next Steps

- Share this dashboard with your NHS team for feedback
- Record a demo video showing the prediction workflow
- Reference this in portfolio interviews
- Use in Session 6 for portfolio packaging

---

**Questions?** Check the governance report for full model documentation.
