import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from pathlib import Path


script_dir = Path(__file__).resolve().parent

repo_dir = script_dir.parent

data_dir = repo_dir / "data"
results_dir = repo_dir / "results"
figure_dir = repo_dir / "figures"

results_dir.mkdir(parents=True, exist_ok=True)
figure_dir.mkdir(parents=True, exist_ok=True)


data_path = data_dir / "vdp_noisy_data.npz"
estimation_path = results_dir / "parameter_estimation_result.npz"

if not data_path.exists():
    raise FileNotFoundError(
        f"Cannot find data file: {data_path}\n"
        "Please run 01_generate_noisy_data.py first."
    )

if not estimation_path.exists():
    raise FileNotFoundError(
        f"Cannot find estimation file: {estimation_path}\n"
        "Please run 02_estimate_parameter.py first."
    )


data = np.load(data_path)
estimation = np.load(estimation_path)

t = data["t"]
x_obs = data["x_obs"]
x_true = data["x_true"]
y_true = data["y_true"]
z0 = data["z0"]

mu_true = float(data["mu_true"])
sigma = float(data["sigma"])
n_points = int(data["n_points"])

mu_hat = float(estimation["mu_hat"])
x_fit = estimation["x_fit"]


def van_der_pol(t, z, mu):
    x, y = z
    dxdt = y
    dydt = mu * (1 - x**2) * y - x
    return [dxdt, dydt]


def simulate_trajectory(mu, t_eval):
    solution = solve_ivp(
        fun=lambda t, z: van_der_pol(t, z, mu),
        t_span=(t_eval[0], t_eval[-1]),
        y0=z0,
        t_eval=t_eval,
        method="RK45",
        rtol=1e-8,
        atol=1e-10
    )

    if not solution.success:
        raise RuntimeError("ODE solver failed: " + solution.message)

    x_model = solution.y[0]
    y_model = solution.y[1]

    return x_model, y_model


def estimate_mu_from_observations(x_observed):
    def residual(mu_array):
        mu = float(mu_array[0])
        x_model, _ = simulate_trajectory(mu, t)
        return x_model - x_observed

    result = least_squares(
        fun=residual,
        x0=np.array([mu_hat]),
        bounds=([0.1], [10.0]),
        xtol=1e-10,
        ftol=1e-10,
        gtol=1e-10
    )

    return float(result.x[0])


rng = np.random.default_rng(seed=123)

n_bootstrap = 100
mu_bootstrap = []

for b in range(n_bootstrap):
    noise_b = rng.normal(loc=0.0, scale=sigma, size=len(x_fit))
    x_boot = x_fit + noise_b

    mu_b = estimate_mu_from_observations(x_boot)
    mu_bootstrap.append(mu_b)

    if (b + 1) % 10 == 0:
        print(f"Finished bootstrap sample {b + 1}/{n_bootstrap}")

mu_bootstrap = np.array(mu_bootstrap)

mu_boot_mean = np.mean(mu_bootstrap)
mu_boot_std = np.std(mu_bootstrap, ddof=1)
mu_ci_lower = np.percentile(mu_bootstrap, 2.5)
mu_ci_upper = np.percentile(mu_bootstrap, 97.5)


plt.figure(figsize=(7, 4.5))

plt.hist(mu_bootstrap, bins=20, edgecolor="black", alpha=0.75)
plt.axvline(mu_true, linestyle=":", linewidth=2, label=f"True $\\mu={mu_true:.2f}$")
plt.axvline(mu_hat, linestyle="--", linewidth=2, label=f"Estimated $\\hat{{\\mu}}={mu_hat:.3f}$")
plt.axvline(mu_ci_lower, linestyle="-.", linewidth=2, label="95% bootstrap interval")
plt.axvline(mu_ci_upper, linestyle="-.", linewidth=2)

plt.xlabel("$\\hat{\\mu}$")
plt.ylabel("Frequency")
plt.title("Parametric bootstrap distribution of estimated $\\mu$")
plt.legend()
plt.tight_layout()

bootstrap_figure_path = figure_dir / "bootstrap_mu_distribution.png"
plt.savefig(bootstrap_figure_path, dpi=300)
plt.show()


t_pred_end = 40.0
n_pred_points = 400
t_pred = np.linspace(t[0], t_pred_end, n_pred_points)

x_true_pred, y_true_pred = simulate_trajectory(mu_true, t_pred)
x_hat_pred, y_hat_pred = simulate_trajectory(mu_hat, t_pred)

bootstrap_trajectories = []

for mu_b in mu_bootstrap:
    x_b, _ = simulate_trajectory(mu_b, t_pred)
    bootstrap_trajectories.append(x_b)

bootstrap_trajectories = np.array(bootstrap_trajectories)

x_pred_median = np.percentile(bootstrap_trajectories, 50, axis=0)
x_pred_lower = np.percentile(bootstrap_trajectories, 5, axis=0)
x_pred_upper = np.percentile(bootstrap_trajectories, 95, axis=0)


plt.figure(figsize=(9, 4.8))

plt.scatter(
    t,
    x_obs,
    s=18,
    alpha=0.55,
    label="Noisy observations"
)

plt.plot(
    t_pred,
    x_true_pred,
    linewidth=2,
    label=f"True trajectory, $\\mu={mu_true:.2f}$"
)

plt.plot(
    t_pred,
    x_pred_median,
    linestyle="--",
    linewidth=2,
    label="Bootstrap median prediction"
)

plt.fill_between(
    t_pred,
    x_pred_lower,
    x_pred_upper,
    alpha=0.25,
    label="90% bootstrap prediction band"
)

plt.axvline(t[-1], linestyle=":", linewidth=2, label="End of observed data")

plt.xlabel("Time")
plt.ylabel("$x(t)$")
plt.title("Prediction band from bootstrap parameter uncertainty")
plt.legend()
plt.tight_layout()

prediction_figure_path = figure_dir / "prediction_band.png"
plt.savefig(prediction_figure_path, dpi=300)
plt.show()


result_path = results_dir / "bootstrap_prediction_result.npz"

np.savez(
    result_path,
    mu_true=mu_true,
    mu_hat=mu_hat,
    sigma=sigma,
    n_points=n_points,
    n_bootstrap=n_bootstrap,
    mu_bootstrap=mu_bootstrap,
    mu_boot_mean=mu_boot_mean,
    mu_boot_std=mu_boot_std,
    mu_ci_lower=mu_ci_lower,
    mu_ci_upper=mu_ci_upper,
    t_pred=t_pred,
    x_true_pred=x_true_pred,
    x_hat_pred=x_hat_pred,
    x_pred_median=x_pred_median,
    x_pred_lower=x_pred_lower,
    x_pred_upper=x_pred_upper,
    bootstrap_trajectories=bootstrap_trajectories
)


summary_path = results_dir / "bootstrap_prediction_summary.txt"

with open(summary_path, "w", encoding="utf-8") as f:
    f.write("Bootstrap and prediction band summary\n")
    f.write("=====================================\n")
    f.write(f"n_points: {n_points}\n")
    f.write(f"Noise level sigma: {sigma:.6f}\n")
    f.write(f"True mu: {mu_true:.6f}\n")
    f.write(f"Estimated mu from original data: {mu_hat:.6f}\n")
    f.write(f"Number of bootstrap samples: {n_bootstrap}\n")
    f.write("\n")
    f.write("Bootstrap parameter uncertainty\n")
    f.write("--------------------------------\n")
    f.write(f"Bootstrap mean: {mu_boot_mean:.6f}\n")
    f.write(f"Bootstrap standard deviation: {mu_boot_std:.6f}\n")
    f.write(f"95% bootstrap interval: [{mu_ci_lower:.6f}, {mu_ci_upper:.6f}]\n")


print("Bootstrap and prediction band finished successfully.")
print(f"n_points: {n_points}")
print(f"sigma: {sigma:.6f}")
print(f"True mu: {mu_true:.6f}")
print(f"Original estimated mu: {mu_hat:.6f}")
print(f"Bootstrap mean mu: {mu_boot_mean:.6f}")
print(f"Bootstrap std mu: {mu_boot_std:.6f}")
print(f"95% bootstrap interval: [{mu_ci_lower:.6f}, {mu_ci_upper:.6f}]")
print(f"Saved bootstrap distribution figure to: {bootstrap_figure_path}")
print(f"Saved prediction band figure to: {prediction_figure_path}")
print(f"Saved numerical results to: {result_path}")
print(f"Saved summary to: {summary_path}")

