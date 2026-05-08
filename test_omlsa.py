import numpy as np

def test_spp(prior_snr, post_snr):
    # Cohen's OMLSA SPP
    v = (prior_snr / (1.0 + prior_snr)) * post_snr
    L = np.exp(np.clip(v, 0, 30)) / (1.0 + prior_snr)
    # q ~ 0.7 => q/(1-q) ~ 2.33
    q = 0.7
    theta = q / (1.0 - q)
    spp = L / (L + theta)
    return spp

prior_arr = np.array([1e-3, 0.1, 1.0, 5.0, 10.0])
post_arr = np.array([1e-3, 1.0, 2.0, 10.0, 20.0])
for pr, po in zip(prior_arr, post_arr):
    print(f"prior_snr={pr:.3f}, post_snr={po:.3f} -> SPP={test_spp(pr, po):.3f}")

