import numpy as np
import scipy.special as sp

def spp(prior_snr, post_snr, q=0.5):
    v = (prior_snr / (1.0 + prior_snr)) * post_snr
    v = np.clip(v, 0, 50)
    L = np.exp(v) / (1.0 + prior_snr)
    return L / (L + q / (1.0 - q))

print("Speech/High SNR:", spp(10, 10))
print("Noise/Low SNR:", spp(0.1, 1))
print("Low prior, high post:", spp(0.1, 5))
print("High prior, low post:", spp(5, 1))
