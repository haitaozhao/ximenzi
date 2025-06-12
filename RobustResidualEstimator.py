import numpy as np

class RobustResidualEstimator:
    def __init__(self, window_size=10, ewma_alpha=0.1, outlier_thresh=3.0):
        self.window_size = window_size
        self.ewma_alpha = ewma_alpha
        self.outlier_thresh = outlier_thresh

        self.residuals = []
        self.ewma_mean = None
        self.ewma_var = None

    def update(self, y_true, y_pred):
        e_t = y_true - y_pred
        self.residuals.append(e_t)
        if len(self.residuals) > self.window_size:
            self.residuals.pop(0)

        # EWMA 更新
        if self.ewma_mean is None:
            self.ewma_mean = e_t
            self.ewma_var = 0.0
        else:
            delta = e_t - self.ewma_mean
            self.ewma_mean += self.ewma_alpha * delta
            self.ewma_var = (1 - self.ewma_alpha) * self.ewma_var + self.ewma_alpha * delta**2

    def get_statistics(self):
        errors = np.array(self.residuals)
        n = len(errors)

        if n == 0:
            return None

        if n >= 15:
            median = np.median(errors)
            mad = np.median(np.abs(errors - median))
            method = "median + MAD"

        elif 5 <= n < 15:
            sorted_e = np.sort(errors)
            trimmed = sorted_e[1:-1] if n > 4 else sorted_e
            median = np.median(trimmed)
            mad = np.median(np.abs(trimmed - median))
            method = "trimmed mean + MAD"

        else:
            median = np.mean(errors)
            mad = np.std(errors)
            method = "mean + std (fallback)"

        return {
            'method': method,
            'median_or_trimmed_mean': median,
            'mad_or_std': mad,
            'ewma_mean': self.ewma_mean,
            'ewma_std': np.sqrt(self.ewma_var)
        }

    def is_outlier(self):
        if len(self.residuals) < 3:
            return False

        stats = self.get_statistics()
        latest_e = self.residuals[-1]
        center = stats['median_or_trimmed_mean']
        scale = stats['mad_or_std'] + 1e-8  # 防止除零
        score = np.abs(latest_e - center) / scale

        return score > self.outlier_thresh

    def get_confidence_interval(self, method="robust", level=0.95):
        stats = self.get_statistics()
        if stats is None:
            return None

        z = 1.96 if level == 0.95 else 2.576 if level == 0.99 else 1.0

        if method == "robust":
            center = stats['median_or_trimmed_mean']
            spread = stats['mad_or_std']
        elif method == "ewma":
            center = stats['ewma_mean']
            spread = stats['ewma_std']
        else:
            raise ValueError("method must be 'robust' or 'ewma'")

        lower = center - z * spread
        upper = center + z * spread

        return (lower, upper)
