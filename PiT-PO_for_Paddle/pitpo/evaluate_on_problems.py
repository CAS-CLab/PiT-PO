from scipy.optimize import least_squares
from joblib import Parallel, delayed
import numpy as np
import os
import time
import multiprocessing


# -------------------- Global parameters --------------------
MAX_NPARAMS = 10
params = [1.0] * MAX_NPARAMS
DECIMAL_PLACES = 3
# Grouped-resampling configuration (tune as needed)
GROUPED_TOTAL_STARTS = 16
GROUPED_N_GROUPS = 8
GROUPED_SUBSET_RATIO = 0.25  # fraction of the original data sampled without replacement per group
MIN_SUBSET_FACTOR = 1.0  # enforce subset length >= n_params * this factor
MIN_UNIFORM_SUBSET = 1000  # if proportional sampling yields too few points, switch to uniform sampling of at least 1000
MAX_UNIFORM_SUBSET = 1500  # if proportional sampling yields too many points (>1500), uniform-sample at most 1500


def _select_parallel_backend() -> str:
    """
    Choose the joblib parallel backend.

    - Main process: default to 'loky' (multiprocess) to fully use CPU cores.
    - Child process (e.g. evaluations launched inside LocalSandbox): force 'threading'
      to avoid "process inside process" issues that cause semlock/folder leaks
      and a flood of shutdown warnings.
    """
    try:
        proc = multiprocessing.current_process()
        if proc is not None and proc.name != "MainProcess":
            return "threading"
    except Exception:
        pass
    return "loky"


# =====================================================
# Complex-step Jacobian + least_squares (LM/TRF)
# =====================================================
def _complex_step_jacobian(
    equation,
    X: np.ndarray,
    outputs: np.ndarray,
    params,
    h: float = 1e-20,
    t0: float | None = None,
    timeout_seconds: float | None = None,
) -> np.ndarray:
    """
    Compute the Jacobian of the residual with respect to the parameters via the complex-step method:
        J[:, i] = d/dp_i [ equation(*X.T, p) - y ].
    Requires that `equation` accepts complex parameters and returns a complex (or complex-castable) array.
    Supports timeout checks inside the loop.
    """
    params = np.asarray(params, dtype=np.float64)
    n = len(params)
    N = outputs.shape[0]
    J = np.empty((N, n), dtype=np.float64)

    base = np.asarray(params, dtype=np.complex128)
    for i in range(n):
        if timeout_seconds is not None and t0 is not None and (
            time.perf_counter() - t0
        ) > timeout_seconds:
            raise TimeoutError("Jacobian timeout")
        p = base.copy()
        p[i] += 1j * h
        yi = equation(*X.T, p)
        J[:, i] = np.imag(yi) / h
    return J


def _run_single_ls(
    equation,
    X: np.ndarray,
    outputs: np.ndarray,
    n_params: int,
    seed: int | None = None,
    method: str = "lm",
    timeout_seconds: float | None = None,
):
    """
    Single non-linear least-squares run.

    The full-data fallback keeps its historical fixed starting point when no
    seed is supplied.  Grouped multi-start evaluation supplies a seed for each
    branch, so those branches use distinct reproducible random starting points.
    Supports LM (unbounded) or TRF (with optional bounds).
    Objective is MSE: res.cost = 0.5 * ||r||^2, so we return mse_ = 2*cost/N.
    """
    if seed is None:
        x0 = np.ones(n_params)
    else:
        rng = np.random.default_rng(seed)
        x0 = rng.uniform(low=-1.0, high=1.0, size=n_params)

    t0 = time.perf_counter()

    def residuals(p):
        if timeout_seconds is not None and (time.perf_counter() - t0) > timeout_seconds:
            raise TimeoutError("Residuals timeout")
        return equation(*X.T, p) - outputs

    def jac(p):
        if timeout_seconds is not None and (time.perf_counter() - t0) > timeout_seconds:
            raise TimeoutError("Jacobian timeout pre-check")
        return _complex_step_jacobian(
            equation, X, outputs, p, t0=t0, timeout_seconds=timeout_seconds
        )

    try:
        if method == "lm":
            res = least_squares(
                residuals,
                x0,
                jac=jac,
                method="lm",
                max_nfev=200,
                # xtol=1e-12,
                # ftol=1e-12,
                # gtol=1e-12,
                xtol=1e-5,
                ftol=1e-5,
                gtol=1e-5,
                verbose=0,
            )
        else:
            res = least_squares(
                residuals,
                x0,
                jac=jac,
                method="trf",
                max_nfev=200,
                xtol=1e-12,
                ftol=1e-12,
                gtol=1e-12,
                verbose=0,
            )

        if res.success and np.isfinite(res.cost):
            mse = 2.0 * res.cost / outputs.size
            res.mse_ = float(mse)
        else:
            res.mse_ = np.inf
        return res
    except Exception:
        class _Fail:
            success = False
            mse_ = np.inf
            x = None

        return _Fail()


def evaluate_least_squares(
    equation,
    X: np.ndarray,
    outputs: np.ndarray,
    n_params: int = 10,
    timeout_seconds: float | None = 10.0,
):
    """
    Least-squares optimization with complex-step Jacobian.
    Tries LM first, then falls back to TRF if LM fails.
    """
    # Try LM first
    result = _run_single_ls(
        equation,
        X,
        outputs,
        n_params,
        method="lm",
        timeout_seconds=timeout_seconds,
    )

    # If LM failed, try TRF
    if (not result.success) or (not np.isfinite(getattr(result, "mse_", np.inf))):
        result = _run_single_ls(
            equation,
            X,
            outputs,
            n_params,
            method="trf",
            timeout_seconds=timeout_seconds,
        )

    if result is None or not np.isfinite(getattr(result, "mse_", np.inf)):
        return None
    return result


# ===================== Grouped-resampling multi-start least_squares =====================
def grouped_subset_multi_start_ls(
    equation: callable,
    X: np.ndarray,
    outputs: np.ndarray,
    n_total_starts: int = GROUPED_TOTAL_STARTS,
    n_groups: int = GROUPED_N_GROUPS,
    subset_ratio: float = GROUPED_SUBSET_RATIO,
    n_params: int = MAX_NPARAMS,
    timeout_seconds: float | None = 10.0,
):
    """Split the total number of starts into groups; for each group, sample a subset of the
    original data (fraction = subset_ratio) and run multi-start least_squares (complex-step
    Jacobian) on it. Pick the parameter vector with the best subset MSE and then compute
    the final MSE on the full data.

    Unified parallelism: all starts across all groups are submitted together, avoiding
    serial group execution.

    Conditions that trigger uniform resampling:
    - Subset too small: proportional sampling yields fewer than MIN_UNIFORM_SUBSET points.
    - Subset too large: proportional sampling yields more than MAX_UNIFORM_SUBSET points.
    Uniform resampling picks min(N, target subset size) points by linear-spaced indices on
    [0, N-1], ensuring the subset is globally uniform.

    Returns dict:
        {
            'best_group': int,
            'subset_best_mse': float,          # best subset MSE across groups
            'final_mse': float,                # MSE on the full data using those params
            'params': list[float] | None,      # best parameters (length n_params)
            'group_records': list[dict]        # per-group records: group, subset_mse, subset_size, params
        }
        or None (on failure).
    """
    if n_total_starts % n_groups != 0:
        raise ValueError("n_total_starts must be divisible by n_groups")
    starts_per_group = n_total_starts // n_groups
    N = outputs.shape[0]

    # Candidate subset size from the configured ratio
    subset_size_candidate = max(int(MAX_NPARAMS * MIN_SUBSET_FACTOR), int(N * subset_ratio), 1)

    # Decide whether to use uniform resampling
    use_uniform = False
    target_uniform_size = None
    if N > 0:
        if subset_size_candidate < MIN_UNIFORM_SUBSET:
            use_uniform = True
            target_uniform_size = min(MIN_UNIFORM_SUBSET, N)
        elif subset_size_candidate > MAX_UNIFORM_SUBSET:
            use_uniform = True
            target_uniform_size = min(MAX_UNIFORM_SUBSET, N)

    if use_uniform:
        subset_size = int(target_uniform_size)
    else:
        subset_size = min(subset_size_candidate, N)

    backend = _select_parallel_backend()

    # Build one subset of indices per group and one seed per start
    group_subsets = []  # [(g, idx_subset)]
    for g in range(n_groups):
        if use_uniform:
            # Uniform resampling: linear-spaced subset_size indices
            idx_subset = np.linspace(0, N - 1, subset_size, dtype=int)
        else:
            # Original logic: random sampling without replacement
            idx_subset = np.random.choice(N, size=subset_size, replace=False)
        group_subsets.append((g, idx_subset))

    seeds = np.random.randint(np.iinfo(np.int32).max, size=n_total_starts)

    # Build every parallel task: each one belongs to a group and uses that group's subset
    tasks = []  # [(g, seed, X_sub, y_sub)]
    for g, idx_subset in group_subsets:
        X_sub = X[idx_subset]
        y_sub = outputs[idx_subset]
        for k in range(starts_per_group):
            seed = seeds[g * starts_per_group + k]
            tasks.append((g, seed, X_sub, y_sub))

    n_jobs = min(n_total_starts, os.cpu_count() or 8)

    # First parallel pass: LM
    with Parallel(n_jobs=n_jobs, backend=backend) as parallel:
        res_list = parallel(
            delayed(_run_single_ls)(
                equation,
                X_sub,
                y_sub,
                n_params,
                seed,
                method="lm",
                timeout_seconds=timeout_seconds,
            )
            for (g, seed, X_sub, y_sub) in tasks
        )

    # If every task failed or produced non-finite cost, retry the whole batch with TRF
    if all((not getattr(r, "success", False)) or (not np.isfinite(getattr(r, "mse_", np.inf))) for r in res_list):
        with Parallel(n_jobs=n_jobs, backend=backend) as parallel:
            res_list = parallel(
                delayed(_run_single_ls)(
                    equation,
                    X_sub,
                    y_sub,
                    n_params,
                    seed,
                    method="trf",
                    timeout_seconds=timeout_seconds,
                )
                for (g, seed, X_sub, y_sub) in tasks
            )

    # Aggregate results by group and keep the best of each
    group_records: list[dict] = []
    # Capture each task's group ID for aggregation
    task_groups = [g for (g, _, _, _) in tasks]

    # Pick the best run within each group
    for g in range(n_groups):
        best_sub = None
        best_idx = None
        for i, r in enumerate(res_list):
            if task_groups[i] != g:
                continue
            if (best_sub is None) or (getattr(r, "mse_", np.inf) < getattr(best_sub, "mse_", np.inf)):
                best_sub = r
                best_idx = i
        if best_sub is not None and np.isfinite(getattr(best_sub, "mse_", np.inf)):
            group_records.append(
                {
                    "group": g,
                    "subset_mse": float(best_sub.mse_),
                    "subset_size": int(subset_size),
                    "params": getattr(best_sub, "x", None),
                    "success": bool(getattr(best_sub, "success", False)),
                    "sampling": "uniform" if use_uniform else "random",
                }
            )
        else:
            group_records.append(
                {
                    "group": g,
                    "subset_mse": np.inf,
                    "subset_size": int(subset_size),
                    "params": None,
                    "success": False,
                    "sampling": "uniform" if use_uniform else "random",
                }
            )

    # Across all groups, pick the one with the smallest subset MSE
    best_group_rec = None
    for rec in group_records:
        if (best_group_rec is None) or (rec["subset_mse"] < best_group_rec["subset_mse"]):
            best_group_rec = rec

    if best_group_rec is None or not np.isfinite(best_group_rec["subset_mse"]) or best_group_rec["params"] is None:
        print(f"[grouped_subset_multi_start_ls] None reason: best_group_rec={best_group_rec}")
        return None

    # Compute final MSE on the full data
    params_best = best_group_rec["params"]
    try:
        y_pred_full = equation(*X.T, params_best)
        final_mse = float(np.mean((y_pred_full - outputs) ** 2))
    except Exception as e:
        print(f"[grouped_subset_multi_start_ls] full-data evaluation error: {e}")
        return None

    return {
        "best_group": int(best_group_rec["group"]),
        "subset_best_mse": float(best_group_rec["subset_mse"]),
        "final_mse": final_mse,
        "params": params_best.tolist() if hasattr(params_best, "tolist") else list(params_best),
        "group_records": group_records,
    }


# ===================== Grouped-resampling evaluation entry point =====================
def evaluate_grouped(data: dict, equation) -> float | None:
    """Grouped-subset multi-start least_squares evaluation; returns the negative final MSE."""
    inputs, outputs = data["inputs"], data["outputs"]
    X = inputs
    result = grouped_subset_multi_start_ls(
        equation,
        X,
        outputs,
        n_total_starts=GROUPED_TOTAL_STARTS,
        n_groups=GROUPED_N_GROUPS,
        subset_ratio=GROUPED_SUBSET_RATIO,
        n_params=MAX_NPARAMS,
        timeout_seconds=10.0,
    )
    if result is None:
        print("[evaluate_grouped] grouped optimization failed; falling back to full-data optimization")
        fallback = evaluate_least_squares(
            equation,
            X,
            outputs,
            n_params=MAX_NPARAMS,
            timeout_seconds=10.0,
        )
        if (
            fallback is None
            or (not getattr(fallback, "success", False))
            or (not np.isfinite(getattr(fallback, "mse_", np.inf)))
        ):
            print(f"[evaluate_grouped] fallback failed: fallback={fallback} success={getattr(fallback,'success',None)} mse_={getattr(fallback,'mse_',None)}")
            return None
        final_mse = float(fallback.mse_)
        if not np.isfinite(final_mse):
            print(f"[evaluate_grouped] fallback None reason: final_mse not finite {final_mse}")
            return None
        if final_mse < 0:
            print(f"[evaluate_grouped] fallback None reason: final_mse<0 {final_mse}")
            return None
        return -final_mse
    final_mse = result["final_mse"]
    if not np.isfinite(final_mse):
        print(f"[evaluate_grouped] None reason: final_mse not finite {final_mse}")
        return None
    if final_mse < 0:
        print(f"[evaluate_grouped] None reason: final_mse<0 {final_mse}")
        return None
    print(f"final_loss={final_mse}")
    return -float(final_mse)


# ===================== Main evaluation entry point =====================
def evaluate(data: dict, equation) -> float | None:
    """
    Main evaluation function called by evaluator.py.
    Uses single-start least_squares optimization with complex-step Jacobian.
    Returns the negative MSE as the score (higher is better).
    """
    inputs, outputs = data["inputs"], data["outputs"]
    X = inputs

    result = evaluate_least_squares(
        equation,
        X,
        outputs,
        n_params=MAX_NPARAMS,
        timeout_seconds=10.0,
    )

    if (
        result is None
        or (not getattr(result, "success", False))
        or (not np.isfinite(getattr(result, "mse_", np.inf)))
    ):
        return None

    final_mse = float(result.mse_)
    if not np.isfinite(final_mse) or final_mse < 0:
        return None

    return -final_mse


def evaluate_with_params(data: dict, equation, use_grouped: bool = True):
    """Evaluate the equation and return both score and parameters; can use grouped-subset acceleration.
    Returns: {"score": float, "optimized_params": list[float] | None, plus grouped-mode metadata}.
    Returns None on failure.
    """
    inputs, outputs = data["inputs"], data["outputs"]
    X = inputs

    # Grouped-subset path
    if use_grouped:
        grouped_res = grouped_subset_multi_start_ls(
            equation,
            X,
            outputs,
            n_total_starts=GROUPED_TOTAL_STARTS,
            n_groups=GROUPED_N_GROUPS,
            subset_ratio=GROUPED_SUBSET_RATIO,
            n_params=MAX_NPARAMS,
            timeout_seconds=10.0,
        )
        if grouped_res is not None:
            final_mse = grouped_res["final_mse"]
            if np.isfinite(final_mse) and final_mse >= 0:
                return {
                    "score": -float(final_mse),
                    "optimized_params": grouped_res["params"],
                    "best_group": grouped_res["best_group"],
                    "subset_best_mse": grouped_res["subset_best_mse"],
                    "group_records": grouped_res["group_records"],
                    "mode": "grouped",
                }
            else:
                print(f"[evaluate_with_params] grouped result invalid MSE final_mse={final_mse}")
        else:
            print("[evaluate_with_params] grouped optimization failed; falling back to full optimization")

    # Fallback: full-data least_squares
    result = evaluate_least_squares(
        equation,
        X,
        outputs,
        n_params=MAX_NPARAMS,
        timeout_seconds=10.0,
    )
    if (
        result is None
        or (not getattr(result, "success", False))
        or (not np.isfinite(getattr(result, "mse_", np.inf)))
    ):
        print(f"[evaluate_with_params] fallback failed result={result} success={getattr(result,'success',None)} mse_={getattr(result,'mse_',None)}")
        return None

    final_loss = float(result.mse_)
    if not np.isfinite(final_loss):
        print(f"[evaluate_with_params] fallback MSE not finite final_loss={final_loss}")
        return None
    if final_loss < 0:
        print(f"[evaluate_with_params] fallback MSE<0 final_loss={final_loss}")
        return None

    try:
        params_out = result.x.tolist() if hasattr(result, "x") else None
    except Exception:
        params_out = None
    print(f"[evaluate_with_params] fallback succeeded final_loss={final_loss} params={params_out}")
    return {
        "score": -final_loss,
        "optimized_params": params_out,
        "best_group": None,
        "subset_best_mse": None,
        "group_records": [],
        "mode": "full",
    }
