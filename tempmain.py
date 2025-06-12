estimator = OnlineResidualEstimator(window_size=50, ewma_alpha=0.1)

# 模拟数据流
for t in range(1000):
    y_true = np.sin(t / 50) + np.random.normal(0, 0.1)
    y_pred = np.sin(t / 50) + np.random.normal(0, 0.05)

    estimator.update(y_true, y_pred)

    if t % 100 == 0:
        stats = estimator.get_robust_statistics()
        print(f"t={t}, Median={stats['median']:.4f}, MAD={stats['mad']:.4f}, EWMA μ={stats['ewma_mean']:.4f}")
