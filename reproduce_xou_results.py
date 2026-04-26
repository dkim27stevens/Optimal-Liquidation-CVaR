"""
Reproduce the XOU stock and European call figures/tables used in the paper.

This is a GitHub-friendly entry point for the XOU numerical results.  It calls
the checked-in XOU helper scripts with the final paper settings:

    stock: gamma = 1.5, lambda = 1.5, Z_t ~= 0.2
    call : gamma = 0.2, lambda = 1.5, Z_t^C ~= 0.2

Both cases use the Leung-style XOU benchmark parameters:

    S0 = 50, T = 0.5, r = 0.03, beta = 4, theta = log(60), eta = 0.3

Run from the repository root:

    python reproduce_xou_results.py

Outputs are written under:

    paper_reproduction_outputs/XOU result/
"""

from pathlib import Path

import generate_xou_call_gamma02_lambda15_plot_and_performance as xou_call
import generate_xou_exposure_cvar_stock_call_results as xou_common
import generate_xou_stock_gamma15_lambda15_performance_table as xou_stock_perf
import generate_xou_stock_gamma15_lambda15_policy_plot_z02 as xou_stock_plot


OUTPUT_DIR = Path.cwd() / "paper_reproduction_outputs" / "XOU result"


def configure_output_dirs() -> None:
    """Redirect all XOU outputs to a portable repository-local folder."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    xou_common.OUTPUT_DIR = OUTPUT_DIR


def main() -> None:
    configure_output_dirs()

    print("Reproducing XOU stock policy-region figure...")
    xou_stock_plot.main()

    print("\nReproducing XOU stock performance table...")
    xou_stock_perf.main()

    print("\nReproducing XOU European call figure and performance table...")
    xou_call.main()

    print(f"\nDone. XOU outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
