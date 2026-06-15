import numpy as np

def fuse_predictions(p_img, p_sym, p_sym_empty=0.00105, tau_img=0.665, w_sym=0.5, cap_up=1.0, eps=1e-6):
    # Mathematical edge case clipping
    p_img = np.clip(p_img, eps, 1.0 - eps)
    p_sym = np.clip(p_sym, eps, 1.0 - eps)
    p_sym_empty = np.clip(p_sym_empty, eps, 1.0 - eps)
    
    # Conversion to log-odds space
    z_img = np.log(p_img / (1.0 - p_img))
    z_sym = np.log(p_sym / (1.0 - p_sym))
    base_logit = np.log(p_sym_empty / (1.0 - p_sym_empty))
    
    # Calculate controlled nudge
    raw_nudge = w_sym * (z_sym - base_logit)
    nudge = float(np.clip(raw_nudge, 0.0, cap_up)) # Only nudge upwards, capped at cap_up
    
    # Re-combine and apply sigmoid activation
    z_fused = z_img + nudge
    p_fused = float(1.0 / (1.0 + np.exp(-z_fused)))
    
    decision = p_fused >= tau_img
    return {
        "p_fused": p_fused,
        "nudge_logodds": nudge,
        "decision": "positive" if decision else "negative"
    }

def run_tests():
    print("--- Running Calibrated Fusion Layer Unit Tests ---")
    
    # Case 1: No symptoms
    res1 = fuse_predictions(p_img=0.500, p_sym=0.00105)
    print(f"Case 1: p_img=0.500, p_sym=0.105% (empty) -> p_fused={res1['p_fused']:.3f}, nudge={res1['nudge_logodds']:.3f}, decision={res1['decision']}")
    assert abs(res1['p_fused'] - 0.500) < 1e-3
    assert abs(res1['nudge_logodds'] - 0.000) < 1e-3
    assert res1['decision'] == 'negative'

    # Case 2: Positive image, no symptoms
    res2 = fuse_predictions(p_img=0.700, p_sym=0.00105)
    print(f"Case 2: p_img=0.700, p_sym=0.105% (empty) -> p_fused={res2['p_fused']:.3f}, nudge={res2['nudge_logodds']:.3f}, decision={res2['decision']}")
    assert abs(res2['p_fused'] - 0.700) < 1e-3
    assert abs(res2['nudge_logodds'] - 0.000) < 1e-3
    assert res2['decision'] == 'positive'

    # Case 3: Marginal image, clinical corroboration (rusty_sputum, etc. -> p_sym=70.5%)
    # Let's see: we want nudge = +1.00.
    # If p_sym = 0.705:
    # z_sym = ln(0.705 / 0.295) = 0.871
    # base_logit = ln(0.00105 / 0.99895) = -6.858
    # raw_nudge = 0.5 * (0.871 - (-6.858)) = 0.5 * 7.729 = 3.864
    # nudge = min(3.864, 1.0) = 1.00
    # z_img = ln(0.55 / 0.45) = 0.20067
    # z_fused = z_img + nudge = 1.20067
    # p_fused = 1 / (1 + exp(-1.20067)) = 0.7686 ≈ 0.769
    res3 = fuse_predictions(p_img=0.550, p_sym=0.705)
    print(f"Case 3: p_img=0.550, p_sym=70.5%00 -> p_fused={res3['p_fused']:.3f}, nudge={res3['nudge_logodds']:.3f}, decision={res3['decision']}")
    assert abs(res3['p_fused'] - 0.769) < 1e-3
    assert abs(res3['nudge_logodds'] - 1.000) < 1e-3
    assert res3['decision'] == 'positive'

    # Case 4: Deep negative image, all 10 symptoms (p_sym=100%)
    res4 = fuse_predictions(p_img=0.050, p_sym=0.999999) # 100% clipped
    print(f"Case 4: p_img=0.050, p_sym=100.0% -> p_fused={res4['p_fused']:.3f}, nudge={res4['nudge_logodds']:.3f}, decision={res4['decision']}")
    # z_img = ln(0.05 / 0.95) = -2.9444
    # nudge = 1.00
    # z_fused = -2.9444 + 1.0 = -1.9444
    # p_fused = 1 / (1 + exp(-(-1.9444))) = 0.125
    assert abs(res4['p_fused'] - 0.125) < 1e-3
    assert abs(res4['nudge_logodds'] - 1.000) < 1e-3
    assert res4['decision'] == 'negative'
    print("SUCCESS: All 4 spec test cases passed successfully!")

if __name__ == "__main__":
    run_tests()
