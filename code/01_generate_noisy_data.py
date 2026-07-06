import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from pathlib import Path


script_dir = Path(__file__).resolve().parent

repo_dir = script_dir.parent

figure_dir = repo_dir / "figures"
data_dir = repo_dir / "data"

figure_dir.mkdir(parents=True, exist_ok=True)
data_dir.mkdir(parents=True, exist_ok=True)


def van_der_pol(t, z, mu):
    x, y = z
    dxdt = y
    dydt = mu * (1 - x**2) * y - x
    return [dxdt, dydt]


mu_true = 2.0
z0 = [2.0, 0.0]

t_start = 0.0
t_end = 10.0

n_points = 60
sigma = 0.4

t_eval = np.linspace(t_start, t_end, n_points)


solution = solve_ivp(
    fun=lambda t, z: van_der_pol(t, z, mu_true),
    t_span=(t_start, t_end),
    y0=z0,
    t_eval=t_eval,
    method="RK45",
    rtol=1e-8,
    atol=1e-10
)

if not solution.success:
    raise RuntimeError("ODE solver failed: " + solution.message)

t = solution.t
x_true = solution.y[0]
y_true = solution.y[1]


rng = np.random.default_rng(seed=42)

x_obs = x_true + rng.normal(
    loc=0.0,
    scale=sigma,
    size=len(x_true)
)


npz_path = data_dir / "vdp_noisy_data.npz"

np.savez(
    npz_path,
    t=t,
    x_true=x_true,
    y_true=y_true,
    x_obs=x_obs,
    mu_true=mu_true,
    sigma=sigma,
    n_points=n_points,
    z0=np.array(z0)
)


csv_path = data_dir / "vdp_noisy_data.csv"

data_table = np.column_stack([
    t,
    x_true,
    y_true,
    x_obs
])

np.savetxt(
    csv_path,
    data_table,
    delimiter=",",
    header="t,x_true,y_true,x_obs",
    comments=""
)


plt.figure(figsize=(8, 4.5))
plt.plot(t, x_true, linewidth=2, label="True trajectory $x(t)$")
plt.scatter(t, x_obs, s=18, alpha=0.65, label="Noisy observations")
plt.xlabel("Time")
plt.ylabel("$x(t)$")
plt.title("Van der Pol oscillator: true trajectory and noisy observations")
plt.legend()
plt.tight_layout()

figure_path_1 = figure_dir / "true_vs_noisy_observations.png"
plt.savefig(figure_path_1, dpi=300)
plt.show()


plt.figure(figsize=(5, 5))
plt.plot(x_true, y_true, linewidth=2)
plt.xlabel("$x(t)$")
plt.ylabel("$y(t)$")
plt.title("Phase portrait of the Van der Pol oscillator")
plt.tight_layout()

figure_path_2 = figure_dir / "phase_portrait.png"
plt.savefig(figure_path_2, dpi=300)
plt.show()


print("Data generation finished successfully.")
print(f"n_points = {n_points}")
print(f"sigma = {sigma}")
print(f"Figures saved in: {figure_dir}")
print(f"Data saved in: {data_dir}")
print(f"Saved NPZ file: {npz_path}")
print(f"Saved CSV file: {csv_path}")

