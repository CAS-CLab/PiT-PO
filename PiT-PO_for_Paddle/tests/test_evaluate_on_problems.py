import numpy as np

from pitpo import evaluate_on_problems


def test_grouped_multi_start_accepts_seeded_branches_and_fits_linear_data():
    x = np.linspace(-2.0, 2.0, 64)
    X = x.reshape(-1, 1)
    outputs = 2.5 * x - 0.75

    def equation(x_values, params):
        return params[0] * x_values + params[1]

    result = evaluate_on_problems.grouped_subset_multi_start_ls(
        equation,
        X,
        outputs,
        n_total_starts=2,
        n_groups=1,
        subset_ratio=1.0,
        n_params=2,
        timeout_seconds=5.0,
    )

    assert result is not None
    assert np.isfinite(result["final_mse"])
    assert result["final_mse"] < 1e-12


def test_full_data_single_start_keeps_fixed_default_initialization():
    x = np.linspace(-1.0, 1.0, 32)
    X = x.reshape(-1, 1)
    outputs = -1.25 * x + 0.5

    def equation(x_values, params):
        return params[0] * x_values + params[1]

    result = evaluate_on_problems._run_single_ls(
        equation,
        X,
        outputs,
        n_params=2,
        method="lm",
        timeout_seconds=5.0,
    )

    assert result.success
    assert result.mse_ < 1e-12
