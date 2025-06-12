est = RobustResidualEstimator(window_size=8)

for t in range(30):
    y_true = np.sin(t / 10) + np.random.normal(0, 0.1)
    y_pred = np.sin(t / 10) + np.random.normal(0, 0.05)
    est.update(y_true, y_pred)

    stats = est.get_statistics()
    is_outlier = est.is_outlier()
    ci_robust = est.get_confidence_interval(method="robust")
    ci_ewma = est.get_confidence_interval(method="ewma")

    print(f"[t={t}] method={stats['method']}, outlier={is_outlier}, "
          f"robust CI={ci_robust}, EWMA CI={ci_ewma}")

