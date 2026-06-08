import os
import math
import numpy as np


# --- Alpha (dNeff) Params ---

zlow_alpha_params_dict = {
      'A0': 0.0705,
      'gamma_A': 0.2467,
      'B0': -0.0303,
      'gamma_B': -4.7636,
      'kt0': 0.2174,
      'gamma_kt': 5.9190,
      'alpha': 0.4346,
      'beta': 0.9912,
}

zhigh_alpha_params_dict = {
      'A0': 0.108,
      'gamma_A': -0.226,
      'B0': -12.390,
      'gamma_B': -28.323,
      'kt0': 0.349,
      'gamma_kt': 1.059,
      'alpha': 0.446,
      'beta': 0.654,
}

# --- Beta (fidm) Params ---
zlow_beta_params_dict = {
      'A0': -8.3800,
      'gamma_A': 3.7346,
      'B0': -0.0211,
      'gamma_B': 0.3233,
      'kt0': 0.0298,
      'gamma_kt': -0.8030,
      'alpha': 1.9431,
      'beta': 2.7706,
}
zhigh_beta_params_dict = {
      'A0': -0.004,
      'gamma_A': 3.484,
      'B0': -0.019,
      'gamma_B': 0.477,
      'kt0': 0.052,
      'gamma_kt': 0.022,
      'alpha': 1.067,
      'beta': 1.399,
}

# --- Baseline Low-z (z <= 0.3) Params ---
lowz_params_dict = {
      'A0': 15.6421,
      'gamma_A': 3.0700,
      'B0': 0.7650,
      'gamma_B': 0.0737,
      'kt0': 0.0437,
      'gamma_kt': -0.5715,
      'alpha': 3.519,
      'beta': 1.1585,
}

# --- Baseline High-z (z > 0.3) Params ---
highz_params_dict = {
      'A0': 36.5656,
      'gamma_A': -4.1392,
      'B0': 0.8117,
      'gamma_B': -0.1534,
      'kt0': 0.0289,
      'gamma_kt': 0.9580,
      'alpha': 2.8759,
      'beta': 0.8457,
}

def spoon_model(k, z, A0, gamma_A, B0, gamma_B, kt0, gamma_kt, alpha, beta):
    u = 1.0 + z
    A_z  = A0  * u**gamma_A   
    B_z  = B0  * u**gamma_B   
    kt_z = kt0 * u**gamma_kt  
    
    f_low = np.exp(-A_z * k**alpha) 
    W = (k**beta) / (kt_z**beta + k**beta)
    return f_low * (1 - W) + B_z * W

def alpha_beta_model(k, z, A0, gamma_A, B0, gamma_B, kt0, gamma_kt, alpha, beta):
    """
    Fits the derivative response function (alpha or beta).
    Starts at 0, dips to a negative minimum, and rises to a positive asymptote.
    """
    u = 1.0 + z
    
    # --- Time-Evolving Parameters ---
    A_z  = A0  * u**gamma_A   # Amplitude of the negative dip
    B_z  = B0  * u**gamma_B   # Positive High-k asymptote
    kt_z = kt0 * u**gamma_kt  # Transition scale (where it crosses 0)
    
    # --- Functional Components ---
    W = (k**beta) / (kt_z**beta + k**beta)
    
    # Notice the negative sign and POSITIVE alpha exponent!
    f_low = -A_z * k**alpha  
    f_high = B_z
    
    return f_low * (1 - W) + f_high * W

def combined_full_model(k, z, dNeff_shift, fidm_shift,):
    # Compute potential alpha and beta from the scaling relations
    alpha = 0
    beta = 0
    if dNeff_shift != 0 and z <= 0.3:
        alpha = alpha_beta_model(k, z, **zlow_alpha_params_dict)
    if fidm_shift != 0 and z <= 0.3:
        beta = alpha_beta_model(k, z, **zlow_beta_params_dict)
    if dNeff_shift != 0 and z > 0.3:
        alpha = alpha_beta_model(k, z, **zhigh_alpha_params_dict)
    if fidm_shift != 0 and z > 0.3:
        beta = alpha_beta_model(k, z, **zhigh_beta_params_dict)
    # Compute baseline contribution
    if z <= 0.3:
        C0 = spoon_model(k, z, **lowz_params_dict)
    else:
        C0 = spoon_model(k, z, **highz_params_dict)

    C = C0 + alpha * dNeff_shift + beta * fidm_shift

    res = {'k': k, 'C': C, 'C0': C0, 'alpha': alpha, 'beta': beta}

    return res