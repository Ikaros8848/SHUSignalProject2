from scipy.special import expi
import numpy as np

def test_omlsa(prior_snr, post_snr):
    v = (prior_snr / (1.0 + prior_snr)) * post_snr
    v_clipped = np.clip(v, 1e-10, 30.0)
    
    gain_lsa = (prior_snr / (1.0 + prior_snr)) * np.exp(-0.5 * expi(-v_clipped))
    
    likelihood = np.exp(v_clipped) / (1.0 + prior_snr)
    q_prob = 0.8
    theta = q_prob / (1.0 - q_prob)
    smooth_speech_presence = likelihood / (likelihood + theta)
    
    speech_threshold = 2.0
    sigmoid_presence = 1.0 / (1.0 + np.exp(-1.4 * (post_snr - speech_threshold)))
    
    speech_presence = smooth_speech_presence * sigmoid_presence
    
    gain_floor = 0.1
    gain = (gain_lsa ** speech_presence) * (gain_floor ** (1.0 - speech_presence))
    
    return gain_lsa, smooth_speech_presence, sigmoid_presence, speech_presence, gain

for po in [0.5, 1.0, 2.0, 5.0, 10.0]:
    pr = po * 0.2
    glsa, spp1, spp2, spp, g = test_omlsa(pr, po)
    print(f"post={po}, prior={pr:.2f} | lsa={glsa:.3f}, spp={spp:.3f} | final_gain={g:.3f}")

