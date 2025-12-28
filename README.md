# 📊 EAPCET Analytics & Rank Simulator

An explainable machine-learning based analytics and simulation system for estimating **TS EAPCET ranks with confidence intervals**, built entirely using **synthetic data** for learning, experimentation, and analysis.

This project prioritizes **interpretability, uncertainty awareness, and end-to-end system design** over black-box prediction.

---

## 🎯 Problem Statement

Students often want to understand:
- How their **score translates into rank**
- How **uncertain** that prediction is
- How they compare against **average and top performers**

Official rank predictors are opaque and not suitable for experimentation.  
This project simulates that ecosystem in a **transparent, configurable, and explainable** manner.

---

## 🧠 System Overview

The system consists of:
1. An **ML-based rank prediction model**
2. An interactive **Streamlit analytics dashboard**
3. A **configuration-driven evaluation system**
4. A **test suite** validating correctness and stability

The project is built as a **single reproducible pipeline**, not disconnected scripts.

---

## 🤖 Machine Learning Model

**File:** `eapcet_rank_model.py`

### Model Design
- **Algorithm:** Gradient Boosting Regressor
- **Prediction Strategy:** Quantile Regression  
  - 10% quantile → optimistic rank  
  - 50% quantile → median (AI Rank)  
  - 90% quantile → pessimistic rank  

### Why This Approach?
- Avoids black-box deep learning
- Provides **confidence intervals**, not just point predictions
- Captures non-linear score → rank relationships
- Robust to noisy, synthetic distributions

### Input Features
- Score
- Year
- Attempted questions
- Correct answers

---

## 📈 Metrics & Interpretation

| Metric | Description |
|------|------------|
| **AI Rank** | Median (50%) predicted rank |
| **Rank Range** | 10% – 90% confidence interval |
| **Percentile** | `100 × (1 − rank / total_candidates)` |
| **Accuracy** | `(correct / attempted) × 100` |
| **Performance Label** | Rule-based classification using percentile |

Performance labels:
- Below Average
- Average
- Good
- Excellent

All thresholds are **fully configurable**.

---

## 🖥️ Streamlit Dashboard

**File:** `eapcet_streamlit_app.py`

### UI Features
- Clean, white, professional UI
- Performance summary cards
- Confidence-aware rank visualization
- Score comparison:
  - Your Score
  - National Average
  - Topper Benchmark
- Color-coded performance indicators
- Clear disclaimers for responsible usage

The dashboard is designed for **analytics and insight**, not marketing visuals.

---

## ⚙️ Configuration System

**File:** `config.py`

Centralized configuration for:
- Performance thresholds
- UI colors and indicators
- Percentile assumptions
- Feature toggles

Allows **rapid experimentation** without modifying model or UI logic.

---

## 🧪 Testing & Tooling

### Test Suite
**File:** `test_system.py`

Validates:
- Model output stability
- Metric calculations
- End-to-end system execution

### Supporting Tools
- Synthetic dataset explorer
- Dataset metadata documentation
- Setup script for quick local execution

## 📁 Project Structure
├── eapcet_rank_model.py        # ML model & training logic
├── eapcet_streamlit_app.py    # Streamlit dashboard
├── eapcet_generator_fixed.py  # Synthetic data generation
├── config.py                  # Central configuration
├── test_system.py             # System tests
├── requirements.txt
├── README.md

🚀 How to Run Locally
git clone https://github.com/MohammedAbdulRehman-2005/eapcet-rank-predictor
cd eapcet-rank-predictor
pip install -r requirements.txt
streamlit run eapcet_streamlit_app.py

⚠️ Known Limitations & Failures (Important)

This section is intentional and honest.

❌ Uses synthetic data only (no official EAPCET data)

❌ Predictions are simulations, not real ranks

❌ No category / reservation modeling

❌ No college or branch cutoff prediction

❌ Rank collisions are allowed (realistic but imperfect)

❌ Assumes a fixed candidate pool size

❌ Does not model inter-year policy changes explicitly

These limitations are acknowledged by design and documented clearly.

🧭 Design Decisions & Trade-offs

Chose explainability over raw accuracy

Avoided deep learning intentionally

Preferred confidence intervals over single values

Designed for learning, experimentation, and analytics, not production deployment

🛡️ Disclaimer

This project is intended only for educational, analytical, and simulation purposes.
It must not be used for official rank prediction or counselling decisions.

👨‍💻 Author

Mohammed Abdul Rehman
B.E. CSE (AI)
Focused on Machine Learning, Analytics, and System Design
