"""
End-to-end GBM stock and European call liquidation results.

This file is self-contained: copy the whole file into a Jupyter notebook cell
or run it as a script.  It reproduces the GBM stock/call policy-region figures
and Monte Carlo performance tables used in the paper.

Required packages: numpy, scipy, pandas, matplotlib
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm


OUTPUT_DIR = Path.cwd() / "paper_reproduction_outputs" / "GBM result"
MC_PATHS = 50_000
MC_SEED = 2026
LAM_CVAR = 1.5
Z_SLICE = 0.8


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


def gbm_transition(s_grid, dt, mu, sigma):
    m = len(s_grid)
    mids = np.empty(m + 1)
    mids[1:-1] = 0.5 * (s_grid[:-1] + s_grid[1:])
    mids[0] = 0.0
    mids[-1] = np.inf
    P = np.empty((m, m))
    drift = (mu - 0.5 * sigma**2) * dt
    vol = sigma * np.sqrt(dt)
    for j, s in enumerate(s_grid):
        mean = np.log(s) + drift
        z_hi = (np.log(mids[1:]) - mean) / vol
        z_lo = (np.log(np.maximum(mids[:-1], 1e-300)) - mean) / vol
        row = norm.cdf(z_hi) - norm.cdf(z_lo)
        row[0] = norm.cdf(z_hi[0])
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


def make_market(params, kind):
    s_grid = np.arange(params["s_min"], params["s_max"] + 1e-12, params["ds"])
    z_grid = np.arange(0.0, params["z_max"] + 1e-12, params["dz"])
    times = np.linspace(0.0, params["T"], params["N"] + 1)
    dt = params["T"] / params["N"]
    P = gbm_transition(s_grid, dt, params["mu"], params["sigma"])

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
                    sigma=params["sigma"],
                )
                for t in times
            ]
        )
        benchmark = float(
            bs_call_price(
                0.0,
                params["S0"],
                K=params["K"],
                T=params["T"],
                r=params["r"],
                sigma=params["sigma"],
            )
        )
        step_exposure = (
            np.exp(-params["r"] * times[:-1, None])
            * np.maximum(benchmark - sale_grid[:-1], 0.0)
            * dt
        )
    else:
        raise ValueError("kind must be 'stock' or 'call'.")

    return {
        "s_grid": s_grid,
        "z_grid": z_grid,
        "times": times,
        "P": P,
        "sale_grid": sale_grid,
        "benchmark": benchmark,
        "step_exposure": step_exposure,
        "dt": dt,
    }


def solve_fixed_y(y, params, lam, market, store_stop=False):
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
        "params": params.copy(),
        "market": market,
        "s_grid": s_grid,
        "z_grid": z_grid,
        "times": times,
        "stop": stop,
        "delta": delta,
    }


def solve_policy(name, params, lam, y_grid):
    market = make_market(params, params["kind"])
    best = None
    for y in y_grid:
        sol = solve_fixed_y(y, params, lam, market, store_stop=False)
        if best is None or sol["outer_value"] > best["outer_value"]:
            best = sol
    best = solve_fixed_y(best["y"], params, lam, market, store_stop=True)
    best["name"] = name
    return best


def extract_lower_boundary_for_z(policy, z_value, min_block=3, allow_left_gap=3):
    s_grid = policy["s_grid"]
    z_grid = policy["z_grid"]
    stop = policy["stop"]
    delta = policy["delta"]
    times = policy["times"]
    z_idx = int(np.argmin(np.abs(z_grid - z_value)))
    actual_z = float(z_grid[z_idx])
    boundary = np.full(len(times), np.nan)
    M = len(s_grid)

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
        d0 = delta[n, j, z_idx]
        d1 = delta[n, j + 1, z_idx]
        if d0 >= 0 and d1 < 0 and abs(d0 - d1) > 1e-12:
            w = d0 / (d0 - d1)
            boundary[n] = s_grid[j] + w * (s_grid[j + 1] - s_grid[j])
        else:
            boundary[n] = s_grid[j]
    return boundary, actual_z


def plot_fixed_z_region(running_policy, cvar_policy, stem, title, z_label):
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), sharex=True, sharey=True)
    y_min, y_max = 50.0, 130.0
    actual_z = Z_SLICE
    for ax, policy, label in [
        (axes[0], running_policy, r"$\lambda=0$"),
        (axes[1], cvar_policy, r"$\lambda=1.5$"),
    ]:
        boundary, actual_z = extract_lower_boundary_for_z(policy, Z_SLICE)
        mask = np.isfinite(boundary)
        t = policy["times"][mask]
        b = boundary[mask]
        ax.fill_between(t, y_min, b, color="#F3C7C7", alpha=0.75, linewidth=0)
        ax.fill_between(t, b, y_max, color="#CFE1F2", alpha=0.85, linewidth=0)
        ax.plot(t, b, color="#C44E52", linewidth=1.8, alpha=0.70)
        ax.text(0.03, b[0] + 8.0, f"start boundary = {b[0]:.2f}", fontsize=9, color="#4D4D4D")
        ax.text(0.55, 121.0, "Delay Region", color="#174A7C", fontsize=11, weight="bold")
        ax.text(0.18, 55.5, "Sell Region", color="#8A1515", fontsize=11, weight="bold")
        ax.set_title(label)
        ax.set_xlabel("Time")
        ax.grid(True, color="#B8B8B8", linewidth=0.7)
        ax.set_xlim(0.0, policy["params"]["T"])
        ax.set_ylim(y_min, y_max)
        if ax is axes[0]:
            ax.set_ylabel("Underlying price")
    fig.suptitle(fr"{title} (${z_label}\approx {actual_z:.1f}$)", y=0.99, fontsize=14)
    fig.tight_layout()
    save_figure(fig, stem)
    plt.show()


def simulate_gbm_paths(params, n_paths=MC_PATHS, seed=MC_SEED):
    rng = np.random.default_rng(seed)
    N = params["N"]
    dt = params["T"] / N
    paths = np.empty((n_paths, N + 1))
    paths[:, 0] = params["S0"]
    shocks = rng.standard_normal((n_paths, N))
    drift = (params["mu"] - 0.5 * params["sigma"] ** 2) * dt
    vol = params["sigma"] * np.sqrt(dt)
    for n in range(N):
        paths[:, n + 1] = paths[:, n] * np.exp(drift + vol * shocks[:, n])
    return paths


def evaluate_policy(policy, paths):
    params = policy["params"]
    market = policy["market"]
    s_grid = policy["s_grid"]
    z_grid = policy["z_grid"]
    times = policy["times"]
    stop = policy["stop"]
    sale_grid = market["sale_grid"]
    benchmark = market["benchmark"]
    dz = params["dz"]
    dt = params["T"] / params["N"]
    n_paths = paths.shape[0]
    tau_idx = np.full(n_paths, params["N"], dtype=int)
    z_acc = np.zeros(n_paths)

    for n in range(params["N"]):
        alive = tau_idx == params["N"]
        if not np.any(alive):
            break
        s_now = paths[alive, n]
        s_idx = np.searchsorted(s_grid, s_now)
        s_idx = np.clip(s_idx, 1, len(s_grid) - 1)
        left = s_grid[s_idx - 1]
        right = s_grid[s_idx]
        s_idx = np.where(np.abs(s_now - left) <= np.abs(s_now - right), s_idx - 1, s_idx)
        z_idx = np.clip(np.rint(z_acc[alive] / dz).astype(int), 0, len(z_grid) - 1)
        should_stop = stop[n, s_idx, z_idx]
        alive_ids = np.where(alive)[0]
        tau_idx[alive_ids[should_stop]] = n

        still_alive = tau_idx == params["N"]
        if np.any(still_alive):
            if params["kind"] == "stock":
                downside = np.maximum(params["S0"] - paths[still_alive, n], 0.0)
            else:
                ids = np.where(still_alive)[0]
                call_now = np.array([np.interp(paths[i, n], s_grid, sale_grid[n]) for i in ids])
                downside = np.maximum(benchmark - call_now, 0.0)
            z_acc[still_alive] += np.exp(-params["r"] * times[n]) * downside * dt

    tau = times[tau_idx]
    s_tau = paths[np.arange(n_paths), tau_idx]
    if params["kind"] == "stock":
        sale = s_tau
    else:
        sale = np.array([np.interp(s_tau[i], s_grid, sale_grid[tau_idx[i]]) for i in range(n_paths)])
    discounted_sale = np.exp(-params["r"] * tau) * sale
    shortfall = np.maximum(benchmark - sale, 0.0)
    ret = (discounted_sale - benchmark) / benchmark
    return {"tau": tau, "sale": sale, "discounted_sale": discounted_sale, "shortfall": shortfall, "return": ret, "exposure": z_acc}


def var_cvar(x, alpha):
    q = np.quantile(x, alpha)
    tail = x[x >= q]
    return q, tail.mean() if len(tail) else q


def summarize(policy, eval_out):
    alpha = policy["params"]["alpha"]
    v_s, c_s = var_cvar(eval_out["shortfall"], alpha)
    v_z, c_z = var_cvar(eval_out["exposure"], alpha)
    row = {
        "Policy": policy["name"],
        "lambda": policy["lam"],
        "best_y": policy["y"],
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
    return row


def write_table(df, stem):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / f"{stem}.csv"
    html_path = OUTPUT_DIR / f"{stem}.html"
    df.to_csv(csv_path, index=False)
    html_path.write_text(df.to_html(index=False, float_format=lambda v: f"{v:.6f}"), encoding="utf-8")
    print(f"Saved table: {csv_path}")
    print(f"Saved table: {html_path}")


def run_case(params, y_grid, fig_stem, table_stem, title, z_label):
    running = solve_policy(
        "Running exposure only" if params["kind"] == "stock" else "Running call exposure only",
        params,
        0.0,
        [0.0],
    )
    cvar = solve_policy(
        "Running + CVaR exposure" if params["kind"] == "stock" else "Running + CVaR call exposure",
        params,
        LAM_CVAR,
        y_grid,
    )
    print(f"{params['kind']} lambda=1.5 best y = {cvar['y']:.3f}")
    plot_fixed_z_region(running, cvar, fig_stem, title, z_label)
    paths = simulate_gbm_paths(params)
    rows = [summarize(p, evaluate_policy(p, paths)) for p in [running, cvar]]
    df = pd.DataFrame(rows)
    write_table(df, table_stem)
    print(df.to_string(index=False))
    return running, cvar, df


def main():
    configure_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    stock_params = dict(
        kind="stock",
        S0=100.0,
        T=1.0,
        r=0.03,
        mu=0.15,
        sigma=0.15,
        alpha=0.95,
        gamma=6.0,
        N=252,
        s_min=40.0,
        s_max=220.0,
        ds=1.0,
        z_max=40.0,
        dz=0.20,
    )
    call_params = dict(
        kind="call",
        S0=100.0,
        K=100.0,
        T=1.0,
        r=0.03,
        mu=0.15,
        sigma=0.15,
        alpha=0.95,
        gamma=6.0,
        N=252,
        s_min=40.0,
        s_max=220.0,
        ds=1.0,
        z_max=12.0,
        dz=0.05,
    )

    run_case(
        stock_params,
        [0.50, 0.80, 1.10, 1.40, 1.70, 2.00],
        "figure_gbm_exposure_cvar_policy_regions_z08_lambda15",
        "performance_table_gbm_stock_lambda15_N252_ds1_dz020",
        "Effect of exposure-CVaR penalty, optimal stock liquidation",
        "Z_t",
    )
    run_case(
        call_params,
        [0.0, 0.30, 0.60, 0.90, 1.20, 1.50],
        "figure_call_exposure_cvar_policy_regions_z08_lambda15",
        "performance_table_gbm_call_lambda15_N252_ds1_dz005",
        "Effect of exposure-CVaR penalty, optimal European call liquidation",
        "Z_t^C",
    )
    print(f"Done. Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
