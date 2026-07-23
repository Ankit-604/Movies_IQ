# 🎬 MovieIQ — Film Predictive Analytics Dashboard

MovieIQ is an interactive, data-driven web dashboard built in Python using Streamlit. The platform leverages a Machine Learning Random Forest Classifier alongside robust statistical diagnostics to explore, analyze, and predict the commercial success of motion picture concepts based on key performance indicators (KPIs) like production budgets, runtime window metrics, popularity indices, and target audience scores.

---

## 🚀 System Architecture & Features

*   **Real-Time Profitability Inference Engine:** An interactive simulator utilizing a trained Random Forest Classifier to dynamically forecast whether a project concept will break even or generate net capital profit based on custom operational constraints.
*   **Exploratory Data Journalism UI:** Clean, custom-styled data distributions, cross-metric comparisons, and interactive data ledgers tracking active records.
*   **Statistical Significance Control:** Embedded diagnostic engines verifying variance factors (via Independent T-Testing) and categorical associations (via Chi-Square Contingency metrics) to differentiate true data trends from random luck.
*   **Granular Sidebar Filtering:** Global multiselect genre filters and numeric sliders to seamlessly segment performance over dynamic subsets of historical data.
*   **Corporate Theming:** Clean, high-end, distraction-free light corporate layout styled intentionally to mimic professional consulting decks rather than stock software.

---

## 📁 Repository Structure

```text
├── .streamlit/
│   └── config.toml      # Custom light corporate branding styles
├── assets/              # Static storage for exported charts and plot graphics
├── MovieIQ.py           # Core Streamlit application & data pipeline code
├── requirements.txt     # Python environment package dependencies
├── movies.csv           # Historical movie ledger tracking performance records
└── README.md            # Project technical documentation
