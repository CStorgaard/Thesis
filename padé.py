import os
import math
import numpy as np


# --- Alpha (dNeff) Params ---
alpha_params_dict = {
  'a1_0': -1.1068,
  'a1_z':  0.2427,
  'a2_0': -24.8801,
  'a2_z':   -1.5559,
  'a3_0': -40.9410,
  'a3_z':   -679.9237,
  'b1_0': 48.1013,
  'b1_z':   0.1129,
  'b2_0': 732.1212,
  'b2_z':   -1.9686,
  'b3_0': 1067.6955,
  'b3_z':   -984.2972
}
# --- Beta (fidm) Params ---
beta_params_dict = {
  'a1_0': 12.2414,
  'a1_z':  -80160.3352,
  'a2_0': 39.4951,
  'a2_z':   17.5683,
  'a3_0': -165374.3809,
  'a3_z':   14.2822,
  'b1_0': 6187.0727,
  'b1_z':   16.7335,
  'b2_0': 393189.6096,
  'b2_z':   -657644.3145,
  'b3_0': 8290461.0047,
  'b3_z':   13.9385
}


# --- Baseline Params ---
params_dict = {
  'a1_0':  16.0794,
  'a1_z':   0.1396,
  'a2_0':   1.8772,
  'a2_z':  -3.5340,
  'a3_0':  54.5102,
  'a3_z':  -2.9914,
  'b1_0':  20.9506,
  'b1_z':   0.1330,
  'b2_0':   6.4875,
  'b2_z':  -0.3864,
  'b3_0':  69.4966,
  'b3_z':  -3.0894
}

# =================================================================
# 3. PADE APPROXIMANT MODEL DEFINITION & BOUNDS
# =================================================================
def pade_model(k, z, a1_0, a1_z, a2_0, a2_z, a3_0, a3_z, b1_0, b1_z, b2_0, b2_z, b3_0, b3_z):
    """
    3rd-Order Padé Approximant.
    Numerator and Denominator coefficients scale as (1+z)^p
    """
    u = 1.0 + z
    
    # Numerator coefficients
    a1 = a1_0 * (u ** a1_z)
    a2 = a2_0 * (u ** a2_z)
    a3 = a3_0 * (u ** a3_z)
    
    # Denominator coefficients
    b1 = b1_0 * (u ** b1_z)
    b2 = b2_0 * (u ** b2_z)
    b3 = b3_0 * (u ** b3_z)
    
    num = 1.0 + a1*k + a2*(k**2) + a3*(k**3)
    den = 1.0 + b1*k + b2*(k**2) + b3*(k**3)
    
    return num / den

def alpha_beta_model(k, z, a1_0, a1_z, a2_0, a2_z, a3_0, a3_z, b1_0, b1_z, b2_0, b2_z, b3_0, b3_z):
    """
    3rd-Order Padé Approximant.
    Numerator and Denominator coefficients scale as (1+z)^p
    """
    u = 1.0 + z
    
    # Numerator coefficients
    a1 = a1_0 * (u ** a1_z)
    a2 = a2_0 * (u ** a2_z)
    a3 = a3_0 * (u ** a3_z)
    
    # Denominator coefficients
    b1 = b1_0 * (u ** b1_z)
    b2 = b2_0 * (u ** b2_z)
    b3 = b3_0 * (u ** b3_z)
    
    num = a1*k + a2*(k**2) + a3*(k**3)
    den = 1.0 + b1*k + b2*(k**2) + b3*(k**3)
    
    return num / den

def combined_full_model(k, z, dNeff_shift, fidm_shift,):
    # Compute potential alpha and beta from the scaling relations
    alpha = 0
    beta = 0
    if dNeff_shift != 0 or fidm_shift != 0:
        alpha = alpha_beta_model(k, z, **alpha_params_dict)
        beta = alpha_beta_model(k, z, **beta_params_dict)
    C0 = pade_model(k, z, **params_dict)

    C = C0 + alpha * dNeff_shift + beta * fidm_shift

    res = {'k': k, 'C': C, 'C0': C0, 'alpha': alpha, 'beta': beta}

    return res