import logging
import numpy as np
from scipy.stats import ks_2samp

# Dummy example of drift detection
def detect_data_drift(reference_data, new_data, threshold=0.05):
    stat, p_value = ks_2samp(reference_data, new_data)
    drift_detected = p_value < threshold
    return drift_detected, p_value

if __name__ == "__main__":
    ref = np.random.normal(0, 1, 1000)
    new = np.random.normal(0.5, 1, 1000)

    drift, p = detect_data_drift(ref, new)
    print(f"Drift Detected: {drift}, p-value: {p}")
