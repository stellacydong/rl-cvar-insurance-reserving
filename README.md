# Adaptive Reserving with CVaR-Constrained RL
**Paper:** https://arxiv.org/abs/2504.09396
**Paper**: [arXiv:2504.09396](https://arxiv.org/abs/2504.09396)

This repository implements the RL-CVaR framework from  
"Adaptive Insurance Reserving with CVaR-Constrained Reinforcement Learning under Macroeconomic Regimes"  
(Stella C. Dong & James R. Finlay, April 2025).

It provides a reproducible pipeline for dynamic reserve optimization using Proximal Policy Optimization (PPO) subject to Conditional Value-at-Risk (CVaR) constraints, trained under a macroeconomic curriculum.

---

## Repository Structure

```
adaptive_reserving_rl/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── wkcomp_pos.csv
│   └── othliab_pos.csv
├── notebooks/
│   ├── rl_cvar_reserving.ipynb
│   └── developing.ipynb
└── src/
    ├── __init__.py
    ├── env.py
    ├── utils.py
    ├── train.py
    └── evaluate.py
```

- **data/**  
  Pre-processed Workers’ Compensation and Other Liability datasets.
- **notebooks/**  
  - `rl_cvar_reserving.ipynb`: end-to-end demo (load, train, evaluate, plot).  
  - `u3.ipynb`: your original notebook.  
- **src/**  
  - `__init__.py`: package initializer.  
  - `env.py`: custom Gymnasium environment for CVaR penalized reserving.  
  - `utils.py`: data loading, preprocessing, metric computation.  
  - `train.py`: PPO training script.  
  - `evaluate.py`: evaluation & fixed-shock stress tests.

---

## Installation

```bash
git clone https://github.com/<your-username>/adaptive_reserving_rl.git
cd adaptive_reserving_rl
pip install -r requirements.txt
```

---

## Usage

### 1. Training

```bash
# Workers’ Compensation
python src/train.py --dataset wkcomp

# Other Liability
python src/train.py --dataset othliab
```

- Trains a PPO agent with CVaR penalization.
- Saves model to `models/ppo_cvar_reserving.zip`.

### 2. Evaluation

```bash
python src/evaluate.py   --model models/ppo_cvar_reserving.zip   --dataset wkcomp
```

- Prints stochastic and fixed-shock metrics.

### 3. Notebook Demo

```bash
jupyter lab notebooks/rl_cvar_reserving.ipynb
```

Walks through:
1. Loading & preprocessing data  
2. Environment instantiation  
3. Model loading/training  
4. Running evaluations  
5. Plotting results  

---

## Configuration

- **Curriculum levels** in `src/train.py`:  
  ```python
  {0: (1.0,0.1), 1: (1.2,0.2), 2: (1.5,0.3), 3: (1.8,0.4)}
  ```
- **Hyperparameters**:  
  `batch_size=2048`, `learning_rate=3e-4`, `n_epochs=10`, `total_timesteps=1e6`.
- **Reward weights** and **buffer size** adjustable in `src/env.py`.

---

## Contributing

Contributions welcome! Please open an issue or pull request.

---

## License

This project is licensed under the MIT License.
