import numpy as np


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))

def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)

def MAE(pred, true):
    return np.mean(np.abs(pred - true))

def MSE(pred, true):
    return np.mean((pred - true) ** 2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def MAPE(pred, true):
    return np.mean(np.abs(100 * (pred - true) / (true +1e-8)))

def MSPE(pred, true):
    return np.mean(np.square((pred - true) / (true + 1e-8)))

def SMAPE(pred, true):
    return np.mean(200 * np.abs(pred - true) / (np.abs(pred) + np.abs(true) + 1e-8))
    # return np.mean(200 * np.abs(pred - true) / (pred + true + 1e-8))

def ND(pred, true):
    return np.mean(np.abs(true - pred)) / np.mean(np.abs(true))

def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    smape = SMAPE(pred, true)
    nd = ND(pred, true)

    return mae, mse, rmse, mape, mspe, smape, nd


# ── 掩码位置指标（用于 Imputation 评估） ──

def masked_mae(pred, true, mask):
    """仅在 mask=True（缺失/需要评估）的位置计算 MAE"""
    error = np.abs(pred - true)
    return error[mask].mean() if mask.sum() > 0 else 0.0


def masked_mse(pred, true, mask):
    """仅在 mask=True 的位置计算 MSE"""
    error = (pred - true) ** 2
    return error[mask].mean() if mask.sum() > 0 else 0.0


def masked_rmse(pred, true, mask):
    """仅在 mask=True 的位置计算 RMSE"""
    return np.sqrt(masked_mse(pred, true, mask))


def masked_mape(pred, true, mask):
    """仅在 mask=True 的位置计算 MAPE"""
    error = np.abs(100 * (pred - true) / (true + 1e-8))
    return error[mask].mean() if mask.sum() > 0 else 0.0


# ── 异常检测指标 ──

def precision_at_k(y_true, y_score, k):
    """Top-k 精度：在分数最高的 k 个点中，真实异常的比例"""
    n = len(y_score)
    k = min(k, n)
    top_k_idx = np.argsort(y_score)[-k:]
    return y_true[top_k_idx].mean()


def recall_at_k(y_true, y_score, k):
    """Top-k 召回：所有真实异常中，被分数最高的 k 个点捕获的比例"""
    n = len(y_score)
    k = min(k, n)
    total_anomalies = y_true.sum()
    if total_anomalies == 0:
        return 0.0
    top_k_idx = np.argsort(y_score)[-k:]
    return y_true[top_k_idx].sum() / total_anomalies


def f1_score(y_true, y_pred):
    """标准二分类 F1"""
    tp = (y_true & y_pred).sum()
    fp = (y_pred & ~y_true).sum()
    fn = (y_true & ~y_pred).sum()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    return 2 * precision * recall / (precision + recall + 1e-8)


def point_adjusted_f1(y_true, y_score, threshold=None):
    """
    Point-adjusted F1：若异常段中任一点被检测到，则整个段被视为检测到。

    这是时序异常检测文献中常用的调整（如 OmniAnomaly, USAD），
    解决了逐点 F1 对异常段边界的过度惩罚。
    """
    if threshold is None:
        threshold = np.percentile(y_score, 95)  # 默认 top-5% 为异常
    y_pred = (y_score >= threshold).astype(bool)

    # 用连通分量检测异常段和预测段
    from scipy.ndimage import label as connected_components
    true_segments, _ = connected_components(y_true)
    pred_segments, _ = connected_components(y_pred)

    # 对每个真实异常段，检查是否有预测点与之重叠
    tp = 0
    for seg_id in range(1, true_segments.max() + 1):
        seg_mask = true_segments == seg_id
        if (pred_segments[seg_mask] > 0).any():
            tp += 1

    fp = max(0, pred_segments.max() - tp)
    fn = max(0, true_segments.max() - tp)

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    return 2 * precision * recall / (precision + recall + 1e-8)
