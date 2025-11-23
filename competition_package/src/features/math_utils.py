import numpy as np


def compute_lagk_autocorr(lag_slice: np.ndarray, k: int) -> np.ndarray:
    """Autocorrelation at lag k (per feature)."""
    if k >= lag_slice.shape[0]:
        return np.zeros(lag_slice.shape[1], dtype=np.float32)
    x = lag_slice[:-k, :]
    y = lag_slice[k:, :]
    x_mean = x.mean(axis=0)
    y_mean = y.mean(axis=0)
    num = ((x - x_mean) * (y - y_mean)).mean(axis=0)
    denom = np.sqrt(((x - x_mean) ** 2).mean(axis=0) * ((y - y_mean) ** 2).mean(axis=0)) + 1e-8
    return (num / denom).astype(np.float32)


def compute_frac_above_mean(lag_slice: np.ndarray, mean_last: np.ndarray) -> np.ndarray:
    return (lag_slice > mean_last[None, :]).mean(axis=0).astype(np.float32)


def compute_robust_window_stats(lag_slice: np.ndarray, mean_last: np.ndarray, std_last: np.ndarray):
    q25 = np.percentile(lag_slice, 25, axis=0).astype(np.float32)
    median = np.percentile(lag_slice, 50, axis=0).astype(np.float32)
    q75 = np.percentile(lag_slice, 75, axis=0).astype(np.float32)
    iqr = (q75 - q25).astype(np.float32)

    denom = std_last + 1e-8
    standardized = (lag_slice - mean_last[None, :]) / denom[None, :]
    skewness = (standardized**3).mean(axis=0).astype(np.float32)
    kurtosis = ((standardized**4).mean(axis=0) - 3.0).astype(np.float32)
    cv = (std_last / (np.abs(mean_last) + 1e-8)).astype(np.float32)
    return q25, median, q75, iqr, skewness, kurtosis, cv


def compute_trend_features(lag_slice: np.ndarray):
    """Slope, R2, curvature over the window."""
    n_lags = lag_slice.shape[0]
    t = np.arange(n_lags, dtype=np.float32)
    sum_t = float(n_lags * (n_lags - 1) / 2.0)
    sum_t2 = float(n_lags * (n_lags - 1) * (2 * n_lags - 1) / 6.0)
    sum_y = lag_slice.sum(axis=0)
    sum_ty = (t[:, None] * lag_slice).sum(axis=0)
    denom = n_lags * sum_t2 - sum_t * sum_t

    if denom == 0.0:
        slope = np.zeros(lag_slice.shape[1], dtype=np.float32)
        r2 = np.zeros(lag_slice.shape[1], dtype=np.float32)
    else:
        slope = (n_lags * sum_ty - sum_t * sum_y) / denom
        intercept = (sum_y - slope * sum_t) / float(n_lags)
        fitted = intercept[None, :] + slope[None, :] * t[:, None]
        residual = lag_slice - fitted
        ss_res = (residual**2).sum(axis=0)
        mean_y = lag_slice.mean(axis=0)
        ss_tot = ((lag_slice - mean_y[None, :]) ** 2).sum(axis=0)
        r2 = 1.0 - ss_res / (ss_tot + 1e-8)

    slope = slope.astype(np.float32)
    r2 = r2.astype(np.float32)

    mid = n_lags // 2
    slope_first = (lag_slice[mid] - lag_slice[0]) / float(mid)
    slope_second = (lag_slice[-1] - lag_slice[mid]) / float(n_lags - mid)
    curvature = (slope_second - slope_first).astype(np.float32)

    return slope, r2, curvature
