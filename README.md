# Bootstrap-Based Uncertainty Propagation for a Nonlinear Dynamical System

This repository contains a compact independent mini-project on uncertainty-aware parameter inference and prediction for a nonlinear dynamical system. The example system is the Van der Pol oscillator. The project estimates an unknown model parameter from noisy partial observations and uses parametric bootstrap to propagate parameter uncertainty into future trajectory prediction bands.

This is a selected public-upload version of the project. It is not a full backup of all exploratory files, and it does not claim empirical calibration or repeated-experiment coverage validation.

## Overview

The final experiment uses noisy observations of only the first state variable, `x(t)`, from a Van der Pol oscillator. The second state variable, `y(t)`, is unobserved. The experiment settings are:

- observation window: `t_end = 10`
- number of observation points: `n_points = 60`
- Gaussian observation noise: `sigma = 0.40`
- true parameter: `mu = 2.0`

The parameter `mu` is estimated by least-squares fitting, then uncertainty in the fitted parameter is quantified with parametric bootstrap and propagated through the ODE model.

## Motivation

Parameter estimates obtained from noisy dynamical-system observations can look precise while still producing uncertain future trajectories. This project demonstrates a simple workflow for connecting parameter uncertainty to prediction uncertainty in a nonlinear ODE setting.

## Mathematical Model

The Van der Pol oscillator is written as

```text
dx/dt = y
dy/dt = mu (1 - x^2) y - x
```

Only noisy measurements of `x(t)` are used for inference:

```text
x_obs(t_i) = x(t_i; mu) + epsilon_i,
epsilon_i ~ N(0, sigma^2)
```

The state variable `y(t)` is latent in the estimation problem.

## Method

1. Generate a synthetic trajectory from the Van der Pol oscillator and observe `x(t)` with Gaussian noise.
2. Estimate `mu` by nonlinear least-squares fitting of model-predicted `x(t)` to noisy observations.
3. Generate parametric bootstrap datasets from the fitted trajectory using the assumed noise level.
4. Re-estimate `mu` for each bootstrap dataset.
5. Propagate bootstrap parameter samples through the ODE model to form prediction bands.

## Results

The final figures are stored in `figures/`:

- [`true_vs_noisy_observations.png`](figures/true_vs_noisy_observations.png): true trajectory and noisy partial observations of `x(t)`.
- [`fitted_trajectory.png`](figures/fitted_trajectory.png): least-squares fitted trajectory compared with the noisy observations.
- [`bootstrap_mu_distribution.png`](figures/bootstrap_mu_distribution.png): bootstrap distribution of the estimated parameter `mu`.
- [`prediction_band.png`](figures/prediction_band.png): future prediction band obtained by propagating bootstrap parameter samples through the nonlinear ODE.

The prediction band widens over the future time interval because small uncertainty in `mu` accumulates into trajectory uncertainty through the nonlinear dynamics.

Compact text summaries are included in `results/`. Large raw numerical arrays are intentionally excluded from the public version.

## Repository Structure

```text
bootstrap-uq-nonlinear-dynamical-system-public-upload/
├── README.md
├── requirements.txt
├── .gitignore
├── code/
│   ├── 01_generate_noisy_data.py
│   ├── 02_estimate_parameter.py
│   └── 03_bootstrap_prediction_band.py
├── data/
│   └── vdp_noisy_data.csv
├── figures/
│   ├── true_vs_noisy_observations.png
│   ├── fitted_trajectory.png
│   ├── bootstrap_mu_distribution.png
│   └── prediction_band.png
└── results/
    ├── parameter_estimation_summary.txt
    └── bootstrap_prediction_summary.txt
```

## How to Run

Install the requirements:

```bash
pip install -r requirements.txt
```

Run the workflow from the repository root:

```bash
python code/01_generate_noisy_data.py
python code/02_estimate_parameter.py
python code/03_bootstrap_prediction_band.py
```

The scripts will regenerate the data, figures, result summaries, and intermediate `.npz` arrays for the final experiment setting. The intermediate `.npz` arrays are ignored by `.gitignore` and are not included in this public-upload folder.

## Requirements

The scripts use:

- NumPy
- SciPy
- Matplotlib

## Limitations and Possible Extensions

This version uses a single synthetic experiment and parametric bootstrap uncertainty propagation. Empirical calibration and repeated-experiment coverage validation are not included. Possible extensions include repeated simulations to assess coverage, comparison with Bayesian inference, joint inference with partially observed states, and robustness studies under misspecified noise assumptions.

## Note

This repository is intended as a concise research-code portfolio project. It does not claim publication, empirical calibration, or coverage validation.


