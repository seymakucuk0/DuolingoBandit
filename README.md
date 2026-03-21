# DuolingoBandit

**Adaptation of Duolingo's Sleeping, Recovering Bandit Algorithm for Optimizing Push Notifications of Kariyer.Net**

Koç University — INDR 491 Engineering Design Project (Spring 2026)

## Overview

This project implements the **Recovering Difference Softmax (RDS)** algorithm, originally proposed by [Yancey & Settles (KDD 2020)](https://doi.org/10.1145/3394486.3403351), and adapts it to optimize push notification delivery for [Kariyer.net](https://kariyer.net), Turkey's largest digital recruitment platform.

### Key Components

- **Relative Difference Scoring** — Measures each notification template's incremental lift over a counterfactual baseline, not raw click rates
- **Recency Penalty** — Exponential decay on recently sent templates to combat notification fatigue
- **Sleeping Arms** — Dynamically filters templates based on user eligibility at decision time
- **Softmax Selection** — Balances exploration of new templates with exploitation of high-performing ones
- **Send-Time Optimization** — Extension beyond the original paper: optimizes *when* to send, not just *what* to send

## Project Structure

```
├── data/                  # Data loading and preprocessing
├── src/                   # Core algorithm implementation
│   ├── bandit/            # Multi-armed bandit framework
│   ├── scoring/           # Relative difference scoring & Bayesian regularization
│   ├── recency/           # Recency penalty (memory decay curve)
│   ├── evaluation/        # Offline policy evaluation (importance sampling)
│   └── send_time/         # Send-time optimization module
├── notebooks/             # Exploratory analysis & experiment notebooks
├── experiments/           # Experiment configs and results
├── tests/                 # Unit and integration tests
└── docs/                  # System design document & reports
```

## Methodology

1. **Validate on Duolingo dataset** (~200M notification events, 34 days) — confirm RDS outperforms random selection
2. **Adapt to Kariyer.net** — redefine reward signals (app open / job listing click), arm definitions (template × user segment), and eligibility rules
3. **Offline evaluation** — weighted importance sampling to estimate policy performance from historical logs
4. **Send-time optimization** — estimate per-user engagement windows and add send time as an arm dimension

## Setup

```bash
# Clone the repository
git clone https://github.com/seymakucuk0/DuolingoBandit.git
cd DuolingoBandit

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Team

| Name | Role |
|------|------|
| Görkem Katayıfçı | Team Member |
| Nazlı Zişan Karademir | Team Member |
| Şeymanur Küçük | Team Member |
| Yücel Ocak | Team Member |

**Academic Supervisor:** Mehmet Gönen

## References

1. K. P. Yancey and B. Settles, "A sleeping, recovering bandit algorithm for optimizing recurring notifications," *KDD '20*, 2020. [doi:10.1145/3394486.3403351](https://doi.org/10.1145/3394486.3403351)
2. A. B. Morrison, M. Misheva, and P. Stoneman, "The effect of timing and frequency of push notifications on users' engagement," *GoodIT '21*, 2021.
3. L. Li, W. Chu, J. Langford, and R. E. Schapire, "A contextual-bandit approach to personalized news article recommendation," *WWW '10*, 2010.
4. A. Slivkins, "Introduction to multi-armed bandits," *Foundations and Trends in Machine Learning*, 2019.
5. N. Bidargaddi et al., "To prompt or not to prompt?," *npj Digital Medicine*, 2020.

## License

This project is developed as part of Koç University INDR 491 coursework.
