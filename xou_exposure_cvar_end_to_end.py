"""
End-to-end XOU stock and European call liquidation results.

This file is self-contained: copy the whole file into a Jupyter notebook cell
or run it as a script.  It reproduces the XOU stock/call policy-region figures
and Monte Carlo performance tables used in the paper.

Required packages: numpy, scipy, pandas, matplotlib
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm


OUTPUT_DIR = Path.cwd() / "paper_reproduction_outputs" / "XOU result"
MC_PATHS = 50_000
MC_SEED = 2026
LAM_CVAR = 1.5
Z_SLICE = 0.2


def configure_style():
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 12,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_figure(fig, stem):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{stem}.eps", dpi=600, bbox_inches="tight")


def make_price_grid(s_min, s_max, ds):
    n = int(round((s_max - s_min) / ds)) + 1
    return np.linspace(s_min, s_max, n)


def xou_transition(s_grid, dt, beta, theta, eta):
    x_grid = np.log(s_grid)
    m = len(s_grid)
    mids = np.empty(m + 1)
    mids[1:-1] = 0.5 * (x_grid[:-1] + x_grid[1:])
    mids[0] = -np.inf
    mids[-1] = np.inf
    decay = np.exp(-beta * dt)
    variance = (eta**2 / (2.0 * beta)) * (1.0 - np.exp(-2.0 * beta * dt))
    std = np.sqrt(variance)
    effective_theta = theta - 0.5 * eta**2 / beta
    P = np.empty((m, m))
    for j, x in enumerate(x_grid):
        mean_x = effective_theta + (x - effective_theta) * decay
        z_hi = (mids[1:] - mean_x) / std
        z_lo = (mids[:-1] - mean_x) / std
        row = norm.cdf(z_hi) - norm.cdf(z_lo)
        row = np.maximum(row, 0.0)
        P[j] = row / row.sum()
    return P


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
    if np.isscalar(t) and t >= T:
        return np.maximum(s - K, 0.0)
    return price


def make_market(params, kind, augmented=True):
    s_grid = make_price_grid(params["s_min"], params["s_max"], params["ds"])
    times = np.linspace(0.0, params["T"], params["N"] + 1)
    dt = params["T"] / params["N"]
    P = xou_transition(s_grid, dt, params["beta"], params["theta"], params["eta"])

    if kind == "stock":
        sale_grid = np.tile(s_grid, (params["N"] + 1, 1))
        benchmark = params["S0"]
        step_exposure = (
            np.exp(-params["r"] * times[:-1, None])
            * np.maximum(params["S0"] - s_grid[None, :], 0.0)
            * dt
        )
    elif kind == "call":
        sale_grid = np.vstack(
            [
                bs_call_price(
                    t,
                    s_grid,
                    K=params["K"],
                    T=params["T"],
                    r=params["r"],
                    sigma=params["eta"],
                )
                for t in times
            ]
        )
        benchmark = float(np.interp(params["S0"], s_grid, sale_grid[0]))
        step_exposure = (
            np.exp(-params["r"] * times[:-1, None])
            * np.maximum(benchmark - sale_grid[:-1], 0.0)
            * dt
        )
    else:
        raise ValueError("kind must be 'stock' or 'call'.")

    market = {
        "s_grid": s_grid,
        "times": times,
        "P": P,
        "sale_grid": sale_grid,
        "benchmark": benchmark,
        "step_exposure": step_exposure,
        "dt": dt,
    }
    if augmented:
        market["z_grid"] = np.arange(0.0, params["z_max"] + 1e-12, params["dz"])
    return market


def solve_running_only_1d(params, kind):
    market = make_market(params, kind, augmented=False)
    s_grid = market["s_grid"]
    times = market["times"]
    P = market["P"]
    sale_grid = market["sale_grid"]
    step_exposure = market["step_exposure"]
    N = params["N"]
    M = len(s_grid)

    V = np.exp(-params["r"] * times[-1]) * sale_grid[-1]
    stop = np.zeros((N + 1, M), dtype=bool)
    delta = np.full((N + 1, M), np.nan, dtype=np.float32)
    stop[N, :] = True
    for n in range(N - 1, -1, -1):
        stop_payoff = np.exp(-params["r"] * times[n]) * sale_grid[n]
        cont = P @ V - params["gamma"] * step_exposure[n]
        do_stop = stop_payoff >= cont
        V = np.where(do_stop, stop_payoff, cont)
        stop[n] = do_stop
        delta[n] = (stop_payoff - cont).astype(np.float32)

    return {
        "name": "Running exposure only" if kind == "stock" else "Running call exposure only",
        "lam": 0.0,
        "y": 0.0,
        "kind": kind,
        "params": params.copy(),
        "market": market,
        "s_grid": s_grid,
        "times": times,
        "stop": stop,
        "delta": delta,
    }


def solve_fixed_y(y, params, lam, market, kind, store_stop=False):
    s_grid = market["s_grid"]
    z_grid = market["z_grid"]
    times = market["times"]
    P = market["P"]
    sale_grid = market["sale_grid"]
    step_exposure = market["step_exposure"]
    dz = params["dz"]
    z_max = z_grid[-1]
    N = params["N"]
    M = len(s_grid)
    Kz = len(z_grid)
    kappa = lam / (1.0 - params["alpha"]) if lam > 0 else 0.0
    payoff_z = -params["gamma"] * z_grid - kappa * np.maximum(z_grid - y, 0.0)
    V = np.exp(-params["r"] * times[-1]) * sale_grid[-1, :, None] + payoff_z[None, :]
    stop = np.zeros((N + 1, M, Kz), dtype=bool) if store_stop else None
    delta = np.full((N + 1, M, Kz), np.nan, dtype=np.float32) if store_stop else None
    if store_stop:
        stop[N, :, :] = True

    for n in range(N - 1, -1, -1):
        stop_payoff = np.exp(-params["r"] * times[n]) * sale_grid[n, :, None] + payoff_z[None, :]
        cont = np.empty_like(V)
        for i in range(M):
            target_z = np.minimum(z_grid + step_exposure[n, i], z_max)
            pos = target_z / dz
            lo = np.clip(np.floor(pos).astype(np.int32), 0, Kz - 1)
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
        "kind": kind,
        "params": params.copy(),
        "market": market,
        "s_grid": s_grid,
        "z_grid": z_grid,
        "times": times,
        "stop": stop,
        "delta": delta,
    }


def solve_cvar_policy(params, kind, y_grid):
    market = make_market(params, kind, augmented=True)
    best = None
    for y in y_grid:
        sol = solve_fixed_y(y, params, LAM_CVAR, market, kind, store_stop=False)
        if best is None or sol["outer_value"] > best["outer_value"]:
            best = sol
    best = solve_fixed_y(best["y"], params, LAM_CVAR, market, kind, store_stop=True)
    best["name"] = "Running + CVaR exposure" if kind == "stock" else "Running + CVaR call exposure"
    return best


def extract_two_sided_1d(policy, min_block=3, allow_left_gap=5, allow_right_gap=2):
    s_grid = policy["s_grid"]
    stop = policy["stop"]
    delta = policy["delta"]
    times = policy["times"]
    M = len(s_grid)
    lower = np.full(len(times), np.nan)
    upper = np.full(len(times), np.nan)
    for n in range(len(times) - 1):
        blocks = _blocks(stop[n])
        lower_block = next((b for b in blocks if b[2] >= min_block and b[0] <= allow_left_gap), None)
        upper_block = next((b for b in reversed(blocks) if b[2] >= min_block and b[1] >= M - 1 - allow_right_gap), None)
        if lower_block is not None:
            _, j, _ = lower_block
            lower[n] = _interp_lower(s_grid, delta[n], j)
        if upper_block is not None:
            i, _, _ = upper_block
            upper[n] = _interp_upper(s_grid, delta[n], i)
    return lower, upper


def extract_two_sided_for_z(policy, z_value, min_block=3, allow_left_gap=5, allow_right_gap=2):
    z_grid = policy["z_grid"]
    z_idx = int(np.argmin(np.abs(z_grid - z_value)))
    actual_z = float(z_grid[z_idx])
    s_grid = policy["s_grid"]
    stop = policy["stop"]
    delta = policy["delta"]
    times = policy["times"]
    M = len(s_grid)
    lower = np.full(len(times), np.nan)
    upper = np.full(len(times), np.nan)
    for n in range(len(times) - 1):
        blocks = _blocks(stop[n, :, z_idx])
        lower_block = next((b for b in blocks if b[2] >= min_block and b[0] <= allow_left_gap), None)
        upper_block = next((b for b in reversed(blocks) if b[2] >= min_block and b[1] >= M - 1 - allow_right_gap), None)
        if lower_block is not None:
            _, j, _ = lower_block
            lower[n] = _interp_lower(s_grid, delta[n, :, z_idx], j)
        if upper_block is not None:
            i, _, _ = upper_block
            upper[n] = _interp_upper(s_grid, delta[n, :, z_idx], i)
    return lower, upper, actual_z


def _blocks(row):
    blocks = []
    i = 0
    M = len(row)
    while i < M:
        if row[i]:
            j = i
            while j + 1 < M and row[j + 1]:
                j += 1
            blocks.append((i, j, j - i + 1))
            i = j + 1
        else:
            i += 1
    return blocks


def _interp_lower(s_grid, delta_row, j):
    if j >= len(s_grid) - 1:
        return s_grid[j]
    d0 = delta_row[j]
    d1 = delta_row[j + 1]
    if d0 >= 0 and d1 < 0 and abs(d0 - d1) > 1e-12:
        w = d0 / (d0 - d1)
        return s_grid[j] + w * (s_grid[j + 1] - s_grid[j])
    return s_grid[j]


def _interp_upper(s_grid, delta_row, i):
    if i <= 0:
        return s_grid[0]
    d0 = delta_row[i - 1]
    d1 = delta_row[i]
    if d0 < 0 and d1 >= 0 and abs(d1 - d0) > 1e-12:
        w = -d0 / (d1 - d0)
        return s_grid[i - 1] + w * (s_grid[i] - s_grid[i - 1])
    return s_grid[i]


def plot_two_sided_regions(running_policy, cvar_policy, stem, title, y_min, y_max):
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), sharex=True, sharey=True)
    actual_z = Z_SLICE
    for ax, policy, label, is_cvar in [
        (axes[0], running_policy, r"$\lambda=0$", False),
        (axes[1], cvar_policy, r"$\lambda=1.5$", True),
    ]:
        if is_cvar:
            lower, upper, actual_z = extract_two_sided_for_z(policy, Z_SLICE)
        else:
            lower, upper = extract_two_sided_1d(policy)
        mask = np.isfinite(lower) | np.isfinite(upper)
        t = policy["times"][mask]
        lo = lower[mask]
        hi = upper[mask]
        delay_low = np.where(np.isfinite(lo), lo, y_min)
        delay_high = np.where(np.isfinite(hi), hi, y_max)
        ax.fill_between(t, delay_low, delay_high, color="#CFE1F2", alpha=0.85, linewidth=0)
        if np.any(np.isfinite(lo)):
            ax.fill_between(t, y_min, lo, where=np.isfinite(lo), color="#F3C7C7", alpha=0.75, linewidth=0)
            ax.plot(t, lo, color="#C44E52", linewidth=1.9, alpha=0.72)
        if np.any(np.isfinite(hi)):
            ax.fill_between(t, hi, y_max, where=np.isfinite(hi), color="#F3C7C7", alpha=0.75, linewidth=0)
            ax.plot(t, hi, color="#C44E52", linewidth=1.9, alpha=0.72)
        if np.isfinite(hi[0]):
            ax.text(0.02, min(hi[0] + 4.0, y_max - 6.0), f"upper start = {hi[0]:.2f}", fontsize=9, color="#4D4D4D")
        if np.isfinite(lo[0]):
            ax.text(0.02, max(lo[0] + 4.0, y_min + 8.0), f"lower start = {lo[0]:.2f}", fontsize=9, color="#4D4D4D")
        ax.text(0.16, y_min + 0.62 * (y_max - y_min), "Delay Region", color="#174A7C", fontsize=11, weight="bold")
        ax.text(0.28, y_min + 0.90 * (y_max - y_min), "Sell Region", color="#8A1515", fontsize=11, weight="bold")
        if np.any(np.isfinite(lo)):
            ax.text(0.18, y_min + 0.08 * (y_max - y_min), "Sell Region", color="#8A1515", fontsize=10, weight="bold")
        ax.set_title(label)
        ax.set_xlabel("Time")
        ax.grid(True, color="#B8B8B8", linewidth=0.7)
        ax.set_xlim(0.0, policy["params"]["T"])
        ax.set_ylim(y_min, y_max)
        if ax is axes[0]:
            ax.set_ylabel("Underlying price")
    z_label = "Z_t^C" if cvar_policy["kind"] == "call" else "Z_t"
    fig.suptitle(fr"{title} (${z_label}\approx {actual_z:.1f}$)", y=0.99, fontsize=14)
    fig.tight_layout()
    save_figure(fig, stem)
    plt.show()


def simulate_xou_paths(params, n_paths=MC_PATHS, seed=MC_SEED):
    rng = np.random.default_rng(seed)
    N = params["N"]
    dt = params["T"] / N
    beta = params["beta"]
    eta = params["eta"]
    decay = np.exp(-beta * dt)
    std = np.sqrt((eta**2 / (2.0 * beta)) * (1.0 - np.exp(-2.0 * beta * dt)))
    effective_theta = params["theta"] - 0.5 * eta**2 / beta
    x = np.empty((n_paths, N + 1))
    x[:, 0] = np.log(params["S0"])
    shocks = rng.standard_normal((n_paths, N))
    for n in range(N):
        x[:, n + 1] = effective_theta + (x[:, n] - effective_theta) * decay + std * shocks[:, n]
    return np.exp(x)


def evaluate_running_1d(policy, paths):
    params = policy["params"]
    market = policy["market"]
    s_grid = policy["s_grid"]
    times = policy["times"]
    stop = policy["stop"]
    sale_grid = market["sale_grid"]
    benchmark = market["benchmark"]
    dt = params["T"] / params["N"]
    n_paths = paths.shape[0]
    tau_idx = np.full(n_paths, params["N"], dtype=int)
    z_acc = np.zeros(n_paths)
    for n in range(params["N"]):
        alive = tau_idx == params["N"]
        if not np.any(alive):
            break
        s_now = paths[alive, n]
        s_idx = nearest_indices(s_grid, s_now)
        should_stop = stop[n, s_idx]
        alive_ids = np.where(alive)[0]
        tau_idx[alive_ids[should_stop]] = n
        still_alive = tau_idx == params["N"]
        if np.any(still_alive):
            if policy["kind"] == "stock":
                downside = np.maximum(params["S0"] - paths[still_alive, n], 0.0)
            else:
                ids = np.where(still_alive)[0]
                call_now = np.array([np.interp(paths[i, n], s_grid, sale_grid[n]) for i in ids])
                downside = np.maximum(benchmark - call_now, 0.0)
            z_acc[still_alive] += np.exp(-params["r"] * times[n]) * downside * dt
    return finish_eval(policy, paths, tau_idx, z_acc)


def evaluate_cvar(policy, paths):
    params = policy["params"]
    market = policy["market"]
    s_grid = policy["s_grid"]
    z_grid = policy["z_grid"]
    times = policy["times"]
    stop = policy["stop"]
    sale_grid = market["sale_grid"]
    benchmark = market["benchmark"]
    dt = params["T"] / params["N"]
    n_paths = paths.shape[0]
    tau_idx = np.full(n_paths, params["N"], dtype=int)
    z_acc = np.zeros(n_paths)
    for n in range(params["N"]):
        alive = tau_idx == params["N"]
        if not np.any(alive):
            break
        s_now = paths[alive, n]
        s_idx = nearest_indices(s_grid, s_now)
        z_idx = np.clip(np.rint(z_acc[alive] / params["dz"]).astype(int), 0, len(z_grid) - 1)
        should_stop = stop[n, s_idx, z_idx]
        alive_ids = np.where(alive)[0]
        tau_idx[alive_ids[should_stop]] = n
        still_alive = tau_idx == params["N"]
        if np.any(still_alive):
            if policy["kind"] == "stock":
                downside = np.maximum(params["S0"] - paths[still_alive, n], 0.0)
            else:
                ids = np.where(still_alive)[0]
                call_now = np.array([np.interp(paths[i, n], s_grid, sale_grid[n]) for i in ids])
                downside = np.maximum(benchmark - call_now, 0.0)
            z_acc[still_alive] += np.exp(-params["r"] * times[n]) * downside * dt
    return finish_eval(policy, paths, tau_idx, z_acc)


def nearest_indices(grid, values):
    idx = np.searchsorted(grid, values)
    idx = np.clip(idx, 1, len(grid) - 1)
    left = grid[idx - 1]
    right = grid[idx]
    return np.where(np.abs(values - left) <= np.abs(values - right), idx - 1, idx)


def finish_eval(policy, paths, tau_idx, z_acc):
    params = policy["params"]
    market = policy["market"]
    s_grid = policy["s_grid"]
    times = policy["times"]
    sale_grid = market["sale_grid"]
    benchmark = market["benchmark"]
    tau = times[tau_idx]
    s_tau = paths[np.arange(paths.shape[0]), tau_idx]
    if policy["kind"] == "stock":
        sale = s_tau
    else:
        sale = np.array([np.interp(s_tau[i], s_grid, sale_grid[tau_idx[i]]) for i in range(paths.shape[0])])
    discounted_sale = np.exp(-params["r"] * tau) * sale
    shortfall = np.maximum(benchmark - sale, 0.0)
    ret = (discounted_sale - benchmark) / benchmark
    return {"tau": tau, "sale": sale, "discounted_sale": discounted_sale, "shortfall": shortfall, "return": ret, "exposure": z_acc}


def var_cvar(x, alpha):
    q = np.quantile(x, alpha)
    tail = x[x >= q]
    return q, tail.mean() if len(tail) else q


def summarize(name, lam, y, params, eval_out):
    v_s, c_s = var_cvar(eval_out["shortfall"], params["alpha"])
    v_z, c_z = var_cvar(eval_out["exposure"], params["alpha"])
    return {
        "Policy": name,
        "lambda": lam,
        "best_y": y,
        "E[sale]": eval_out["sale"].mean(),
        "E[discounted sale]": eval_out["discounted_sale"].mean(),
        "E[return]": eval_out["return"].mean(),
        "P(shortfall > 0)": np.mean(eval_out["shortfall"] > 1e-12),
        "E[shortfall]": eval_out["shortfall"].mean(),
        "VaR_0.95(shortfall)": v_s,
        "CVaR_0.95(shortfall)": c_s,
        "E[exposure]": eval_out["exposure"].mean(),
        "VaR_0.95(exposure)": v_z,
        "CVaR_0.95(exposure)": c_z,
        "Q99(exposure)": np.quantile(eval_out["exposure"], 0.99),
        "E[tau]": eval_out["tau"].mean(),
        "Median tau": np.median(eval_out["tau"]),
    }


def write_table(df, stem):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / f"{stem}.csv"
    html_path = OUTPUT_DIR / f"{stem}.html"
    df.to_csv(csv_path, index=False)
    html_path.write_text(df.to_html(index=False, float_format=lambda v: f"{v:.6f}"), encoding="utf-8")
    print(f"Saved table: {csv_path}")
    print(f"Saved table: {html_path}")


def run_case(params, kind, y_grid, fig_stem, table_stem, title, y_min, y_max):
    running_params = params.copy()
    if kind == "stock":
        # Match the final paper output: the Leung-style stock benchmark was
        # solved on a finer one-dimensional price grid, while the augmented
        # exposure-CVaR policy used ds=0.25.
        running_params["ds"] = 0.10
    running = solve_running_only_1d(running_params, kind)
    cvar = solve_cvar_policy(params, kind, y_grid)
    print(f"{kind} lambda=1.5 best y = {cvar['y']:.3f}")
    plot_two_sided_regions(running, cvar, fig_stem, title, y_min, y_max)
    paths = simulate_xou_paths(params)
    eval_running = evaluate_running_1d(running, paths)
    eval_cvar = evaluate_cvar(cvar, paths)
    df = pd.DataFrame(
        [
            summarize(running["name"], 0.0, 0.0, params, eval_running),
            summarize(cvar["name"], LAM_CVAR, cvar["y"], params, eval_cvar),
        ]
    )
    write_table(df, table_stem)
    print(df.to_string(index=False))
    return running, cvar, df


def main():
    configure_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    base = dict(
        S0=50.0,
        K=50.0,
        T=0.5,
        r=0.03,
        beta=4.0,
        theta=np.log(60.0),
        eta=0.30,
        alpha=0.95,
        N=252,
    )
    stock_params = dict(
        **base,
        gamma=1.5,
        s_min=0.05,
        s_max=90.0,
        ds=0.25,
        z_max=40.0,
        dz=0.20,
    )
    call_params = dict(
        **base,
        gamma=0.2,
        s_min=0.05,
        s_max=100.0,
        ds=0.25,
        z_max=12.0,
        dz=0.05,
    )

    run_case(
        stock_params,
        "stock",
        [0.0, 0.25, 0.50, 0.80, 1.10, 1.40, 1.70, 2.00],
        "figure_xou_stock_gamma15_lambda0_vs_lambda15_z02",
        "performance_table_xou_stock_gamma15_lambda0_vs_lambda15_leung_benchmark",
        "Effect of exposure-CVaR penalty, optimal XOU stock liquidation",
        0.0,
        90.0,
    )
    run_case(
        call_params,
        "call",
        [0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.80, 1.10, 1.50],
        "figure_xou_call_gamma02_lambda0_vs_lambda15_z02",
        "performance_table_xou_call_gamma02_lambda0_vs_lambda15_leung_benchmark",
        "Effect of exposure-CVaR penalty, optimal XOU European call liquidation",
        -10.0,
        100.0,
    )
    print(f"Done. Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
