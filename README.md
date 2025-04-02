# RL-CVaR-Insurance-Reserving

**Adaptive Insurance Reserving with CVaR-Constrained Reinforcement Learning under Macroeconomic Regimes**

This repository provides a reproducible implementation of the RL-CVaR reserving framework described in the paper:

> *"Adaptive Insurance Reserving with CVaR-Constrained Reinforcement Learning under Macroeconomic Regimes"*

## 🧠 Overview

This framework combines **Reinforcement Learning (RL)** with **Conditional Value-at-Risk (CVaR)** optimization, regime-aware **curriculum learning**, and macro-shock **stress testing** to dynamically allocate insurance reserves. It is applied to long-tail liability lines using real-world datasets from the CAS Loss Reserving Database.

Key features:
- PPO-based RL agent for dynamic reserve allocation  
- Tail-risk penalization using CVaR  
- Macroeconomic volatility modeling and stress testing  
- Curriculum learning across calm, moderate, and recession regimes  
- Benchmarks against CLM, BFM, and stochastic bootstrap

## 📂 Repository Structure


## 📊 Datasets

- **Workers Compensation** and **Other Liability** lines of business  
- Source: [CAS Loss Reserving Database](https://www.casact.org/research/)

> Datasets should be preprocessed using the `load_data()` function in `utils.py`.

## 📈 Reproducibility

The model can be trained via:

```bash
python train.py
