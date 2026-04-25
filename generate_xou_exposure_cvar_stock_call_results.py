from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm


OUTPUT_DIR = Path.cwd() / "paper_plots_XOU_exposure_CVaR_stock_call"

BASE_PARAMS = dict(
    S0=50.0,
    K=50.0,
    T=0.5,
    r=0.03,
    ou_kappa=4.0,
    theta=np.log(60.0),
    eta=0.30,
    alpha=0.95,
    gamma=1.5,
    N=252,
    s_min=1.0,
    s_max=140.0,
    ds=1.0,
    grid_type="log",
    xou_convention="log_ou",
)

LAM_CVAR = 0.6
Z_SLICE = 0.8
MC_PATHS = 50_000
MC_SEED = 2026

STOCK_Y_GRID = np.array([0.00, 0.25, 0.50, 0.80, 1.10, 1.40, 1.70, 2.00])
CALL_Y_GRID = np.array([0.00, 0.10, 0.20, 0.30, 0.50, 0.80, 1.10])

STOCK_Z_MAX = 40.0
STOCK_DZ = 0.20
CALL_Z_MAX = 12.0
CALL_DZ = 0.05

PNG_DPI = 300
PDF_DPI = 300
EPS_DPI = 600


def configure_style():
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 12,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 9,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_figure(fig, stem):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=PNG_DPI, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf", dpi=PDF_DPI, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{stem}.eps", dpi=EPS_DPI, bbox_inches="tight")


def make_price_grid(s_min, s_max, ds=1.0, grid_type="log"):
    if s_min <= 0:
        raise ValueError("s_min must be positive.")
    if s_max <= s_min:
        raise ValueError("s_max must be greater than s_min.")
    if ds <= 0:
        raise ValueError("ds must be positive.")

    approx_nodes = int(round((s_max - s_min) / ds)) + 1
    approx_nodes = max(approx_nodes, 3)
    if grid_type == "linear":
        return np.linspace(s_min, s_max, approx_nodes)
    if grid_type == "log":
        return np.exp(np.linspace(np.log(s_min), np.log(s_max), approx_nodes))
    raise ValueError("grid_type must be 'linear' or 'log'.")


def build_xou_transition(s_grid, dt, ou_kappa, theta, eta, convention="log_ou"):
    s_grid = np.asarray(s_grid, dtype=float)
    if np.any(s_grid <= 0):
        raise ValueError("s_grid must be positive.")
    if np.any(np.diff(s_grid) <= 0):
        raise ValueError("s_grid must be strictly increasing.")
    if dt <= 0:
        raise ValueError("dt must be positive.")
    if eta <= 0:
        raise ValueError("eta must be positive.")
    if ou_kappa < 0:
        raise ValueError("ou_kappa must be nonnegative.")
    if convention not in {"log_ou", "price_exp_ou"}:
        raise ValueError("convention must be 'log_ou' or 'price_exp_ou'.")

    x_grid = np.log(s_grid)
    m = len(s_grid)
    x_mids = np.empty(m + 1)
    x_mids[1:-1] = 0.5 * (x_grid[:-1] + x_grid[1:])
    x_mids[0] = -np.inf
    x_mids[-1] = np.inf

    if ou_kappa > 0:
        decay = np.exp(-ou_kappa * dt)
        variance = (eta**2 / (2.0 * ou_kappa)) * (1.0 - np.exp(-2.0 * ou_kappa * dt))
    else:
        decay = 1.0
        variance = eta**2 * dt
    std = np.sqrt(variance)

    effective_theta = theta
    if convention == "price_exp_ou" and ou_kappa > 0:
        effective_theta = theta - 0.5 * eta**2 / ou_kappa

    P = np.empty((m, m))
    for j, x in enumerate(x_grid):
        mean_x = effective_theta + (x - effective_theta) * decay
        z_hi = (x_mids[1:] - mean_x) / std
        z_lo = (x_mids[:-1] - mean_x) / std
        row = norm.cdf(z_hi) - norm.cdf(z_lo)
        row = np.maximum(row, 0.0)
        P[j] = row / row.sum()
    return P


def make_xou_market(params, *, kind):
    T = params["T"]
    r = params["r"]
    N = params["N"]
    dt = T / N
    s_grid = make_price_grid(
        params["s_min"], params["s_max"], ds=params["ds"], grid_type=params["grid_type"]
    )
    z_max = STOCK_Z_MAX if kind == "stock" else CALL_Z_MAX
    dz = STOCK_DZ if kind == "stock" else CALL_DZ
    z_grid = np.arange(0.0, z_max + 1e-12, dz)
    times = np.linspace(0.0, T, N + 1)
    P = build_xou_transition(
        s_grid,
        dt,
        params["ou_kappa"],
        params["theta"],
        params["eta"],
        convention=params["xou_convention"],
    )

    market = {
        "s_grid": s_grid,
        "z_grid": z_grid,
        "times": times,
        "P": P,
        "dt": dt,
        "dz": dz,
        "z_max": z_max,
    }

    if kind == "stock":
        step_exposure = (
            np.exp(-r * times[:-1, None])
            * np.maximum(params["S0"] - s_grid[None, :], 0.0)
            * dt
        )
        market["sale_grid"] = np.tile(s_grid, (N + 1, 1))
        market["benchmark"] = params["S0"]
        market["step_exposure"] = step_exposure
        return market

    call_grid = price_xou_call_grid(params, s_grid, times, P)
    C0 = float(np.interp(params["S0"], s_grid, call_grid[0]))
    step_exposure = (
        np.exp(-r * times[:-1, None])
        * np.maximum(C0 - call_grid[:-1], 0.0)
        * dt
    )
    market["sale_grid"] = call_grid
    market["benchmark"] = C0
    market["step_exposure"] = step_exposure
    market["C0"] = C0
    return market


def price_xou_call_grid(params, s_grid, times, P_q):
    K = params["K"]
    T = params["T"]
    r = params["r"]
    sigma = params["eta"]
    return np.vstack(
        [bs_call_price(t, s_grid, K=K, T=T, r=r, sigma=sigma) for t in times]
    )


def bs_call_price(t, s, *, K, T, r, sigma):
    s = np.asarray(s, dtype=float)
    tau = np.maximum(T - t, 0.0)
    if np.isscalar(tau) and tau <= 1e-14:
        return np.maximum(s - K, 0.0)
    tau = np.maximum(tau, 1e-14)
    d1 = (np.log(np.maximum(s, 1e-300) / K) + (r + 0.5 * sigma**2) * tau) / (
        sigma * np.sqrt(tau)
    )
    d2 = d1 - sigma * np.sqrt(tau)
    price = s * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)
    if np.isscalar(t):
        if t >= T:
            return np.maximum(s - K, 0.0)
    return price


def solve_fixed_y(y, params, lam, market, *, store_stop=False):
    s_grid = market["s_grid"]
    z_grid = market["z_grid"]
    times = market["times"]
    P = market["P"]
    sale_grid = market["sale_grid"]
    step_exposure = market["step_exposure"]
    dz = market["dz"]
    z_max = market["z_max"]

    r = params["r"]
    T = params["T"]
    alpha = params["alpha"]
    gamma = params["gamma"]
    N = params["N"]
    M = len(s_grid)
    Kz = len(z_grid)

    kappa_cvar = lam / (1.0 - alpha) if lam > 0.0 else 0.0
    payoff_z = -gamma * z_grid - kappa_cvar * np.maximum(z_grid - y, 0.0)
    V = np.exp(-r * T) * sale_grid[-1, :, None] + payoff_z[None, :]

    stop = None
    delta = None
    if store_stop:
        stop = np.zeros((N + 1, M, Kz), dtype=bool)
        stop[N, :, :] = True
        delta = np.full((N + 1, M, Kz), np.nan, dtype=np.float32)

    for n in range(N - 1, -1, -1):
        stop_payoff = np.exp(-r * times[n]) * sale_grid[n, :, None] + payoff_z[None, :]
        cont = np.empty_like(V)

        for i in range(M):
            target_z = np.minimum(z_grid + step_exposure[n, i], z_max)
            pos = target_z / dz
            lo = np.floor(pos).astype(np.int32)
            lo = np.clip(lo, 0, Kz - 1)
            hi = np.minimum(lo + 1, Kz - 1)
            w = pos - lo
            vals = (1.0 - w)[None, :] * V[:, lo] + w[None, :] * V[:, hi]
            cont[i, :] = P[i] @ vals

        do_stop = stop_payoff >= cont
        V = np.where(do_stop, stop_payoff, cont)

        if store_stop:
            stop[n, :, :] = do_stop
            delta[n, :, :] = (stop_payoff - cont).astype(np.float32)

    value0 = float(np.interp(params["S0"], s_grid, V[:, 0]))
    return {
        "value0": value0,
        "outer_value": value0 - lam * y,
        "y": float(y),
        "lam": lam,
        "s_grid": s_grid,
        "z_grid": z_grid,
        "times": times,
        "stop": stop,
        "delta": delta,
        "params": params.copy(),
        "market": market,
    }


def solve_policy(name, params, lam, y_grid, *, kind):
    market = make_xou_market(params, kind=kind)
    if lam == 0.0:
        sol = solve_fixed_y(0.0, params, lam, market, store_stop=True)
        sol["name"] = name
        sol["search_rows"] = [(0.0, sol["value0"], sol["outer_value"])]
        return sol

    search_rows = []
    best_y = None
    best_outer = -np.inf
    for y in y_grid:
        sol = solve_fixed_y(float(y), params, lam, market, store_stop=False)
        search_rows.append((float(y), sol["value0"], sol["outer_value"]))
        if sol["outer_value"] > best_outer:
            best_outer = sol["outer_value"]
            best_y = float(y)

    sol = solve_fixed_y(best_y, params, lam, market, store_stop=True)
    sol["name"] = name
    sol["search_rows"] = search_rows
    return sol


def extract_lower_boundary_for_z(policy, z_value, min_block=3, allow_left_gap=2):
    s_grid = policy["s_grid"]
    z_grid = policy["z_grid"]
    stop = policy["stop"]
    delta = policy["delta"]
    times = policy["times"]
    dz = policy["market"]["dz"]
    z_idx = int(np.clip(round(z_value / dz), 0, len(z_grid) - 1))
    actual_z = z_grid[z_idx]
    M = len(s_grid)
    boundary = np.full(len(times), np.nan)

    for n in range(len(times) - 1):
        row = stop[n, :, z_idx]
        blocks = []
        i = 0
        while i < M:
            if row[i]:
                j = i
                while j + 1 < M and row[j + 1]:
                    j += 1
                blocks.append((i, j, j - i + 1))
                i = j + 1
            else:
                i += 1

        chosen = None
        for i, j, length in blocks:
            if length >= min_block and i <= allow_left_gap:
                chosen = (i, j)
                break
        if chosen is None:
            continue

        _, j = chosen
        if j >= M - 1:
            boundary[n] = s_grid[j]
            continue

        delta_j = delta[n, j, z_idx]
        delta_j1 = delta[n, j + 1, z_idx]
        if delta_j >= 0.0 and delta_j1 < 0.0 and abs(delta_j - delta_j1) > 1e-12:
            w = delta_j / (delta_j - delta_j1)
            boundary[n] = s_grid[j] + w * (s_grid[j + 1] - s_grid[j])
        else:
            boundary[n] = s_grid[j]

    return boundary, actual_z


def extract_two_sided_boundaries_for_z(
    policy, z_value, min_block=3, allow_left_gap=5, allow_right_gap=2
):
    s_grid = policy["s_grid"]
    z_grid = policy["z_grid"]
    stop = policy["stop"]
    delta = policy["delta"]
    times = policy["times"]
    dz = policy["market"]["dz"]
    z_idx = int(np.clip(round(z_value / dz), 0, len(z_grid) - 1))
    actual_z = z_grid[z_idx]
    M = len(s_grid)

    lower = np.full(len(times), np.nan)
    upper = np.full(len(times), np.nan)

    for n in range(len(times) - 1):
        row = stop[n, :, z_idx]
        blocks = []
        i = 0
        while i < M:
            if row[i]:
                j = i
                while j + 1 < M and row[j + 1]:
                    j += 1
                blocks.append((i, j, j - i + 1))
                i = j + 1
            else:
                i += 1

        lower_block = None
        upper_block = None
        for i, j, length in blocks:
            if length >= min_block and i <= allow_left_gap:
                lower_block = (i, j)
                break
        for i, j, length in reversed(blocks):
            if length >= min_block and j >= M - 1 - allow_right_gap:
                upper_block = (i, j)
                break

        if lower_block is not None:
            _, j = lower_block
            if j >= M - 1:
                lower[n] = s_grid[j]
            else:
                delta_j = delta[n, j, z_idx]
                delta_j1 = delta[n, j + 1, z_idx]
                if delta_j >= 0.0 and delta_j1 < 0.0 and abs(delta_j - delta_j1) > 1e-12:
                    w = delta_j / (delta_j - delta_j1)
                    lower[n] = s_grid[j] + w * (s_grid[j + 1] - s_grid[j])
                else:
                    lower[n] = s_grid[j]

        if upper_block is not None:
            i, _ = upper_block
            if i <= 0:
                upper[n] = s_grid[0]
            else:
                delta_im1 = delta[n, i - 1, z_idx]
                delta_i = delta[n, i, z_idx]
                if delta_im1 < 0.0 and delta_i >= 0.0 and abs(delta_i - delta_im1) > 1e-12:
                    w = -delta_im1 / (delta_i - delta_im1)
                    upper[n] = s_grid[i - 1] + w * (s_grid[i] - s_grid[i - 1])
                else:
                    upper[n] = s_grid[i]

    return lower, upper, actual_z


def plot_fixed_z_regions(running_policy, cvar_policy, stem, title, z_label):
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), sharex=True, sharey=True)
    for ax, policy, panel_title in [
        (axes[0], running_policy, r"$\lambda=0$"),
        (axes[1], cvar_policy, rf"$\lambda={LAM_CVAR:g}$"),
    ]:
        lower, upper, actual_z = extract_two_sided_boundaries_for_z(policy, Z_SLICE)
        mask = np.isfinite(lower) | np.isfinite(upper)
        plot_times = policy["times"][mask]
        plot_lower = lower[mask]
        plot_upper = upper[mask]
        y_min = 0.0
        y_max = 90.0

        delay_low = np.where(np.isfinite(plot_lower), plot_lower, y_min)
        delay_high = np.where(np.isfinite(plot_upper), plot_upper, y_max)
        ax.fill_between(plot_times, delay_low, delay_high, color="#CFE1F2", alpha=0.85, linewidth=0)
        if np.any(np.isfinite(plot_lower)):
            ax.fill_between(plot_times, y_min, plot_lower, where=np.isfinite(plot_lower), color="#F3C7C7", alpha=0.75, linewidth=0)
            ax.plot(plot_times, plot_lower, color="#C44E52", linewidth=1.8, alpha=0.65)
        if np.any(np.isfinite(plot_upper)):
            ax.fill_between(plot_times, plot_upper, y_max, where=np.isfinite(plot_upper), color="#F3C7C7", alpha=0.75, linewidth=0)
            ax.plot(plot_times, plot_upper, color="#C44E52", linewidth=1.8, alpha=0.65)

        if np.isfinite(plot_lower[0]):
            ax.text(
                0.03 * BASE_PARAMS["T"],
                max(plot_lower[0] + 3.0, y_min + 3.0),
                f"lower start = {plot_lower[0]:.2f}",
                fontsize=9,
                color="#4D4D4D",
            )
        if np.isfinite(plot_upper[0]):
            ax.text(
                0.03 * BASE_PARAMS["T"],
                min(plot_upper[0] + 3.0, y_max - 5.0),
                f"upper start = {plot_upper[0]:.2f}",
                fontsize=9,
                color="#4D4D4D",
            )
        if policy is cvar_policy:
            ax.text(0.03 * BASE_PARAMS["T"], 12.0, fr"$y^*={policy['y']:.2f}$", fontsize=9, color="#4D4D4D")
        ax.text(0.12 * BASE_PARAMS["T"], 45.0, "Delay Region", color="#174A7C", fontsize=11, weight="bold")
        ax.text(0.55 * BASE_PARAMS["T"], 82.0, "Sell Region", color="#8A1515", fontsize=11, weight="bold")
        ax.set_title(panel_title)
        ax.set_xlabel("Time")
        ax.grid(True, color="#B8B8B8", linewidth=0.7)
        ax.set_xlim(0.0, BASE_PARAMS["T"])
        ax.set_ylim(y_min, y_max)
        if ax is axes[0]:
            ax.set_ylabel("Underlying price")

    fig.suptitle(fr"{title} (${z_label}\approx {actual_z:.1f}$)", y=0.99, fontsize=14)
    fig.tight_layout()
    save_figure(fig, stem)
    plt.close(fig)


def simulate_xou_paths(params, n_paths=MC_PATHS, seed=MC_SEED):
    rng = np.random.default_rng(seed)
    S0 = params["S0"]
    T = params["T"]
    N = params["N"]
    dt = T / N
    beta = params["ou_kappa"]
    theta = params["theta"]
    eta = params["eta"]
    decay = np.exp(-beta * dt)
    if beta > 0:
        std = np.sqrt((eta**2 / (2.0 * beta)) * (1.0 - np.exp(-2.0 * beta * dt)))
    else:
        std = eta * np.sqrt(dt)
    x = np.empty((n_paths, N + 1))
    x[:, 0] = np.log(S0)
    shocks = rng.standard_normal((n_paths, N))
    for n in range(N):
        x[:, n + 1] = theta + (x[:, n] - theta) * decay + std * shocks[:, n]
    return np.exp(x)


def evaluate_policy(policy, paths, *, kind):
    params = policy["params"]
    market = policy["market"]
    S0 = params["S0"]
    T = params["T"]
    r = params["r"]
    N = params["N"]
    dt = T / N
    times = policy["times"]
    s_grid = policy["s_grid"]
    z_grid = policy["z_grid"]
    stop = policy["stop"]
    dz = market["dz"]
    n_paths = paths.shape[0]

    tau_idx = np.full(n_paths, N, dtype=int)
    z_acc = np.zeros(n_paths)

    if kind == "call":
        call_grid = market["sale_grid"]
        C0 = market["benchmark"]

    for n in range(N):
        alive = tau_idx == N
        if not np.any(alive):
            break

        s_now = paths[alive, n]
        s_idx = np.searchsorted(s_grid, s_now)
        s_idx = np.clip(s_idx, 1, len(s_grid) - 1)
        left = s_grid[s_idx - 1]
        right = s_grid[s_idx]
        s_idx = np.where(np.abs(s_now - left) <= np.abs(s_now - right), s_idx - 1, s_idx)
        z_idx = np.rint(z_acc[alive] / dz).astype(int)
        z_idx = np.clip(z_idx, 0, len(z_grid) - 1)

        should_stop = stop[n, s_idx, z_idx]
        alive_indices = np.where(alive)[0]
        tau_idx[alive_indices[should_stop]] = n

        still_alive = tau_idx == N
        if np.any(still_alive):
            if kind == "stock":
                downside = np.maximum(S0 - paths[still_alive, n], 0.0)
            else:
                call_now = np.array(
                    [np.interp(paths[i, n], s_grid, call_grid[n]) for i in np.where(still_alive)[0]]
                )
                downside = np.maximum(C0 - call_now, 0.0)
            z_acc[still_alive] += np.exp(-r * times[n]) * downside * dt

    tau = times[tau_idx]
    s_tau = paths[np.arange(n_paths), tau_idx]
    if kind == "stock":
        sale_value = s_tau
        benchmark = S0
    else:
        call_grid = market["sale_grid"]
        sale_value = np.array([np.interp(s_tau[i], s_grid, call_grid[tau_idx[i]]) for i in range(n_paths)])
        benchmark = market["benchmark"]

    discounted_sale = np.exp(-r * tau) * sale_value
    shortfall = np.maximum(benchmark - sale_value, 0.0)
    discounted_return = (discounted_sale - benchmark) / benchmark
    return {
        "tau": tau,
        "sale_value": sale_value,
        "discounted_sale": discounted_sale,
        "shortfall": shortfall,
        "return": discounted_return,
        "exposure": z_acc,
    }


def var_cvar(x, alpha=0.95):
    q = np.quantile(x, alpha)
    tail = x[x >= q]
    return q, tail.mean() if len(tail) else q


def summarize(policy, eval_out):
    alpha = policy["params"]["alpha"]
    shortfall_var, shortfall_cvar = var_cvar(eval_out["shortfall"], alpha)
    exposure_var, exposure_cvar = var_cvar(eval_out["exposure"], alpha)
    returns = eval_out["return"]
    downside_returns = returns[returns < 0.0]
    downside_std = downside_returns.std(ddof=1) if len(downside_returns) > 1 else np.nan
    return {
        "Policy": policy["name"],
        "lambda": policy["lam"],
        "best_y": policy["y"],
        "E[sale]": eval_out["sale_value"].mean(),
        "E[discounted sale]": eval_out["discounted_sale"].mean(),
        "E[return]": returns.mean(),
        "Std(return)": returns.std(ddof=1),
        "Sharpe": returns.mean() / returns.std(ddof=1),
        "Sortino": returns.mean() / downside_std if np.isfinite(downside_std) and downside_std > 0 else np.nan,
        "P(shortfall > 0)": np.mean(eval_out["shortfall"] > 0.0),
        "E[shortfall]": eval_out["shortfall"].mean(),
        "VaR_0.95(shortfall)": shortfall_var,
        "CVaR_0.95(shortfall)": shortfall_cvar,
        "E[exposure]": eval_out["exposure"].mean(),
        "VaR_0.95(exposure)": exposure_var,
        "CVaR_0.95(exposure)": exposure_cvar,
        "Q99(exposure)": np.quantile(eval_out["exposure"], 0.99),
        "E[tau]": eval_out["tau"].mean(),
        "Median tau": np.median(eval_out["tau"]),
    }


def write_performance_table(policies, paths, *, kind, stem):
    rows = []
    for policy in policies:
        print(f"  evaluating {kind}: {policy['name']}")
        rows.append(summarize(policy, evaluate_policy(policy, paths, kind=kind)))
    df = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_DIR / f"{stem}.csv", index=False)
    html = df.to_html(index=False, float_format=lambda x: f"{x:.6f}")
    (OUTPUT_DIR / f"{stem}.html").write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<style>body{font-family:Arial,sans-serif;margin:24px}table{border-collapse:collapse}"
        "td,th{border:1px solid #ddd;padding:6px 8px;text-align:right}th{background:#f4f4f4}"
        "td:first-child,th:first-child{text-align:left}</style></head><body>"
        f"<h2>{stem}</h2>{html}</body></html>",
        encoding="utf-8",
    )
    return df


def main():
    configure_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    stock_params = BASE_PARAMS.copy()
    stock_params.update(z_max=STOCK_Z_MAX, dz=STOCK_DZ)
    call_params = BASE_PARAMS.copy()
    call_params.update(z_max=CALL_Z_MAX, dz=CALL_DZ)

    print("Solving XOU stock policies...")
    running_stock = solve_policy("Mean exposure penalty only", stock_params, 0.0, STOCK_Y_GRID, kind="stock")
    cvar_stock = solve_policy("Mean + CVaR exposure penalty", stock_params, LAM_CVAR, STOCK_Y_GRID, kind="stock")
    plot_fixed_z_regions(
        running_stock,
        cvar_stock,
        "figure_xou_stock_exposure_cvar_policy_regions_leung_params_z08_lambda06",
        "Effect of exposure-CVaR penalty, optimal XOU stock liquidation",
        "Z_t",
    )
    print(f"  stock best y = {cvar_stock['y']:.3f}")

    print("Solving XOU call policies...")
    running_call = solve_policy("Mean call-exposure penalty only", call_params, 0.0, CALL_Y_GRID, kind="call")
    cvar_call = solve_policy("Mean + CVaR call-exposure penalty", call_params, LAM_CVAR, CALL_Y_GRID, kind="call")
    plot_fixed_z_regions(
        running_call,
        cvar_call,
        "figure_xou_call_exposure_cvar_policy_regions_leung_params_z08_lambda06",
        "Effect of exposure-CVaR penalty, optimal XOU European call liquidation",
        "Z_t^C",
    )
    print(f"  call best y = {cvar_call['y']:.3f}")

    print("Simulating XOU paths and writing performance tables...")
    paths = simulate_xou_paths(BASE_PARAMS)
    stock_df = write_performance_table(
        [running_stock, cvar_stock],
        paths,
        kind="stock",
        stem="performance_table_xou_stock_leung_params_N252_ds1_z08_lambda06",
    )
    call_df = write_performance_table(
        [running_call, cvar_call],
        paths,
        kind="call",
        stem="performance_table_xou_call_leung_params_N252_ds1_z08_lambda06",
    )
    print(stock_df.to_string(index=False))
    print(call_df.to_string(index=False))
    print(f"Saved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
