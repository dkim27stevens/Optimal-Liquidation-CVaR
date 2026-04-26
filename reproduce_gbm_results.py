"""
Reproduce the GBM stock and European call figures/tables used in the paper.

This is a GitHub-friendly entry point for the GBM numerical results.  It calls
the checked-in GBM helper scripts with the final paper settings:

    stock: lambda = 1.5, Z_t ~= 0.8, N = 252, ds = 1, dz = 0.20
    call : lambda = 1.5, Z_t^C ~= 0.8, N = 252, ds = 1, dz = 0.05

Run from the repository root:

    python reproduce_gbm_results.py

Outputs are written under:

    paper_reproduction_outputs/GBM result/
"""

from pathlib import Path

import generate_fixed_z08_lambda15_region_plots as gbm_plots
import generate_gbm_call_exposure_cvar_section52_plots as gbm_call
import generate_gbm_exposure_cvar_section51_plots as gbm_stock
import save_gbm_call_lambda15_performance_table as gbm_call_perf
import save_gbm_stock_lambda15_performance_table as gbm_stock_perf


OUTPUT_DIR = Path.cwd() / "paper_reproduction_outputs" / "GBM result"


def configure_output_dirs() -> None:
    """Redirect all GBM outputs to a portable repository-local folder."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    gbm_stock.OUTPUT_DIR = OUTPUT_DIR
    gbm_call.OUTPUT_DIR = OUTPUT_DIR

    gbm_stock_perf.FINAL_RESULT_DIR = OUTPUT_DIR
    gbm_call_perf.FINAL_RESULT_DIR = OUTPUT_DIR


def main() -> None:
    configure_output_dirs()

    print("Reproducing GBM policy-region figures...")
    gbm_plots.main()

    print("\nReproducing GBM stock performance table...")
    gbm_stock_perf.main()

    print("\nReproducing GBM European call performance table...")
    gbm_call_perf.main()

    print(f"\nDone. GBM outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
