import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon, t

def compare_errors(true, est1, est2, stat_test="ttest_rel", alternative="two-sided",
                   weights=None, groupby=None, agg="mean"):

    if stat_test not in ["ttest_rel", "wilcoxon"]:
        raise ValueError("stat_test must be 'ttest_rel' or 'wilcoxon'")
    else:
        stat_test = ttest_rel if stat_test == "ttest_rel" else wilcoxon
    if alternative not in ["two-sided", "greater", "less"]:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")
    if true.shape[0] != est1.shape[0] or true.shape[0] != est2.shape[0]:
        raise ValueError("true, est1, and est2 must have the same number of observations")

    true, err1, err2, weights = _errors_prep(true, est1, est2, weights=weights, groupby=groupby, agg=agg)
    results = {}
    if ~np.all(weights == 1):
        comparison = ttest_rel_w(err1, err2, popmean=0, weights=weights, alternative=alternative)
        results["stat"], results["pval"] = comparison["statistic"], comparison["pvalue"]
    else:
        comparison = stat_test(err1, err2, alternative=alternative, nan_policy="omit")
        results["stat"], results["pval"] = comparison.statistic, comparison.pvalue
    
    results["mae1"] = np.nansum(err1 * weights, axis=0) / np.nansum(weights, axis=0)
    results["rmse1"] = np.sqrt(np.nansum((err1**2) * weights, axis=0) / np.nansum(weights, axis=0))
    results["mape1"] = np.abs(np.nansum((err1 / true) * weights, axis=0) / np.nansum(weights, axis=0))

    results["mae2"] = np.nansum(err2 * weights, axis=0) / np.nansum(weights, axis=0)
    results["rmse2"] = np.sqrt(np.nansum((err2**2) * weights, axis=0) / np.nansum(weights, axis=0))
    results["mape2"] = np.abs(np.nansum((err2 / true) * weights, axis=0) / np.nansum(weights, axis=0))

    return {k: np.squeeze(v) for k, v in results.items()}


def _errors_prep(true, est1, est2, weights=None, groupby=None, agg="mean"):
    # Reshape to 2D if 1D
    true = true.reshape(-1,1) if true.ndim==1 else true
    est1 = est1.reshape(-1,1) if est1.ndim==1 else est1
    est2 = est2.reshape(-1,1) if est2.ndim==1 else est2
    if weights is not None:
        weights = weights.reshape(-1,1) if weights.ndim==1 else weights
    else:
        weights = np.ones([true.shape[0], 1])

    err1 = np.abs(true - est1)
    err2 = np.abs(true - est2)

    # Group by if specified to test grand mean errors
    if groupby is not None:
        df_tmp  = pd.DataFrame(true)
        true = df_tmp.groupby(groupby).agg(agg).to_numpy()
        df1_tmp = pd.DataFrame(err1)
        err1 = df1_tmp.groupby(groupby).agg(agg).to_numpy()
        df2_tmp = pd.DataFrame(err2)
        err2 = df2_tmp.groupby(groupby).agg(agg).to_numpy()
        df3_tmp = pd.DataFrame(weights)
        weights = df3_tmp.groupby(groupby).agg(agg).to_numpy()

    return true, err1, err2, weights


def ttest_rel_w(a, b, popmean=0, weights=None, alternative="two-sided", axis=0):
    # Difference
    x = a - b  # shape (..., n)
    
    if weights is None:
        weights = np.ones_like(x)
    
    # Weighted mean along last axis
    xw = np.nansum(weights * x, axis=axis) / np.nansum(weights, axis=axis)
    
    # Weighted variance (unbiased)
    s2w = np.nansum(weights * (x - np.expand_dims(xw, axis=axis))**2, axis=axis) / (
        np.nansum(weights, axis=axis) - np.nansum(weights**2, axis=axis) / np.nansum(weights, axis=axis)
    )
    
    # Effective sample size
    neff = (np.nansum(weights, axis=axis))**2 / np.nansum(weights**2, axis=axis)

    # t-statistic
    t_stat = (xw - popmean) / np.sqrt(s2w / neff)
    
    # Degrees of freedom
    df = neff - 1
    
    # p-values
    if alternative == "greater":
        p_value = t.sf(t_stat, df)
    elif alternative == "less":
        p_value = t.cdf(t_stat, df)
    else:  # two-sided
        p_value = 2 * t.sf(np.abs(t_stat), df)
    
    return {"statistic": t_stat, "pvalue": p_value}
