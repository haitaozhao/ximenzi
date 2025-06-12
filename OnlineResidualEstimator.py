import numpy as np


class OnlineResidualEstimator:
    def __init__(self, window_size=100, ewma_alpha=0.05):
        self.window_size = window_size
        self.ewma_alpha = ewma_alpha

        # 初始化历史残差
        self.residuals = []

        # EWMA 均值与方差估计
        self.ewma_mean = None
        self.ewma_var = None

    def update(self, y_true, y_pred):
        e_t = y_true - y_pred
        self.residuals.append(e_t)
        if len(self.residuals) > self.window_size:
            self.residuals.pop(0)

        # 更新 EWMA 均值
        if self.ewma_mean is None:
            self.ewma_mean = e_t
            self.ewma_var = 0.0
        else:
            delta = e_t - self.ewma_mean
            self.ewma_mean += self.ewma_alpha * delta
            self.ewma_var = (1 - self.ewma_alpha) * self.ewma_var + self.ewma_alpha * delta ** 2

    def get_robust_statistics(self):
        if len(self.residuals) == 0:
            return None

        # 中位数和 MAD
        residuals_np = np.array(self.residuals)
        median = np.median(residuals_np)
        mad = np.median(np.abs(residuals_np - median))

        # EWMA 均值和标准差
        ewma_std = np.sqrt(self.ewma_var) if self.ewma_var is not None else None

        return {
            'median': median,
            'mad': mad,
            'ewma_mean': self.ewma_mean,
            'ewma_std': ewma_std
        }
