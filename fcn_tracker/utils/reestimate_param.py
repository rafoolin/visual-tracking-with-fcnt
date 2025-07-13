def reestimate_param(pf_param):
    """
    Re-estimates the particle filter parameters based on confidence and image resolution.

    Args:
        pf_param (TrackerParams.PFParam): Particle filter parameter object.

    Returns:
        pf_param (TrackerParams.PFParam): Possibly updated parameters.
    """
    minconf = pf_param.minconf
    ratio = pf_param.ratio

    # Decision logic for scaling
    if minconf > 0.49:
        scale = False
    elif 0.45 < minconf <= 0.49:
        scale = True
    elif 0.4 < minconf <= 0.45 and ratio > 0.6:
        scale = True
    elif 0.35 < minconf <= 0.4 and ratio < 0.3:
        scale = True
    else:
        scale = False

    if scale:
        pf_param.affsig = pf_param.affsig_o.copy()

    return pf_param
