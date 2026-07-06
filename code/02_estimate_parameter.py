import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from pathlib import Path


script_dir = Path(__file__).resolve().parent

repo_dir = script_dir.parent

data_dir = repo_dir / "data"
figure_dir = repo_dir / "figures"
results_dir = repo_dir / "results"

figure_dir.mkdir(parents=True, exist_ok=True)
results_dir.mkdir(parents=True, exist_ok=True)


data_path = data_dir / "vdp_noisy_data.npz"

if not data_path.exists():
    raise FileNotFoundError(
        f"Cannot find data file: {data_path}\n"
        "Please run 01_generate_noisy_data.py first."
    )

data = np.load(data_path)

t = data["t"]
x_true = data["x_true"]
y_true = data["y_true"]
x_obs = data["x_obs"]
mu_true = float(data["mu_true"])
sigma = float(data["sigma"])
n_points = int(data["n_points"])
z0 = data["z0"]


def van_der_pol(t, z, mu):
    x, y = z
    dxdt = y
    dydt = mu * (1 - x**2) * y - x
    return [dxdt, dydt]


def simulate_trajectory(mu):
    solution = solve_ivp(
        fun=lambda t, z: van_der_pol(t, z, mu),
        t_span=(t[0], t[-1]),
        y0=z0,
        t_eval=t,
        method="RK45",
        rtol=1e-8,
        atol=1e-10
    )

    if not solution.success:
        raise RuntimeError("ODE solver failed: " + solution.message)

    x_model = solution.y[0]
    y_model = solution.y[1]

    return x_model, y_model


def residual(mu_array):
    mu = float(mu_array[0])
    x_model, _ = simulate_trajectory(mu)
    return x_model - x_obs


initial_guess = np.array([1.0])

result = least_squares(
    fun=residual,
    x0=initial_guess,
    bounds=([0.1], [10.0]),
    xtol=1e-10,
    ftol=1e-10,
    gtol=1e-10
)

mu_hat = float(result.x[0])

x_fit, y_fit = simulate_trajectory(mu_hat)

absolute_error = abs(mu_hat - mu_true)
sum_squared_error = np.sum((x_fit - x_obs) ** 2)
root_mean_squared_error = np.sqrt(np.mean((x_fit - x_obs) ** 2))


result_path = results_dir / "parameter_estimation_result.npz"

np.savez(
    result_path,
    mu_true=mu_true,
    mu_hat=mu_hat,
    sigma=sigma,
    n_points=n_points,
    absolute_error=absolute_error,
    sum_squared_error=sum_squared_error,
    root_mean_squared_error=root_mean_squared_error,
    t=t,
    x_true=x_true,
    y_true=y_true,
    x_obs=x_obs,
    x_fit=x_fit,
    y_fit=y_fit
)


summary_path = results_dir / "parameter_estimation_summary.txt"

with open(summary_path, "w", encoding="utf-8") as f:
    f.write("Parameter estimation summary\n")
    f.write("============================\n")
    f.write(f"n_points: {n_points}\n")
    f.write(f"Noise level sigma: {sigma:.6f}\n")
    f.write(f"True mu: {mu_true:.6f}\n")
    f.write(f"Estimated mu: {mu_hat:.6f}\n")
    f.write(f"Absolute error: {absolute_error:.6f}\n")
    f.write(f"Sum of squared errors: {sum_squared_error:.6f}\n")
    f.write(f"Root mean squared error: {root_mean_squared_error:.6f}\n")
    f.write(f"Optimization success: {result.success}\n")
    f.write(f"Optimization message: {result.message}\n")


plt.figure(figsize=(8, 4.5))

plt.plot(
    t,
    x_true,
    linewidth=2,
    label=f"True trajectory, $\\mu={mu_true:.2f}$"
)

plt.scatter(
    t,
    x_obs,
    s=18,
    alpha=0.65,
    label="Noisy observations"
)

plt.plot(
    t,
    x_fit,
    linestyle="--",
    linewidth=2,
    label=f"Fitted trajectory, $\\hat{{\\mu}}={mu_hat:.3f}$"
)

plt.xlabel("Time")
plt.ylabel("$x(t)$")
plt.title("Parameter estimation from noisy observations")
plt.legend()
plt.tight_layout()

fitted_figure_path = figure_dir / "fitted_trajectory.png"
plt.savefig(fitted_figure_path, dpi=300)
plt.show()


mu_grid = np.linspace(0.5, 4.0, 100)
loss_values = []

for mu in mu_grid:
    x_model, _ = simulate_trajectory(mu)
    loss = np.sum((x_model - x_obs) ** 2)
    loss_values.append(loss)

loss_values = np.array(loss_values)

plt.figure(figsize=(7, 4.5))

plt.plot(mu_grid, loss_values, linewidth=2)
plt.axvline(mu_true, linestyle=":", linewidth=2, label=f"True $\\mu={mu_true:.2f}$")
plt.axvline(mu_hat, linestyle="--", linewidth=2, label=f"Estimated $\\hat{{\\mu}}={mu_hat:.3f}$")

plt.xlabel("$\\mu$")
plt.ylabel("Least-squares loss")
plt.title("Loss curve for parameter estimation")
plt.legend()
plt.tight_layout()

loss_figure_path = figure_dir / "loss_curve.png"
plt.savefig(loss_figure_path, dpi=300)
plt.show()


print("Parameter estimation finished successfully.")
print(f"n_points: {n_points}")
print(f"sigma: {sigma:.6f}")
print(f"True mu: {mu_true:.6f}")
print(f"Estimated mu: {mu_hat:.6f}")
print(f"Absolute error: {absolute_error:.6f}")
print(f"RMSE: {root_mean_squared_error:.6f}")
print(f"Saved fitted trajectory figure to: {fitted_figure_path}")
print(f"Saved loss curve figure to: {loss_figure_path}")
print(f"Saved result file to: {result_path}")
print(f"Saved summary file to: {summary_path}")

