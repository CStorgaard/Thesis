import numpy as np

def compute_matter_power_spectrum(cosmo, z, nonlinear=False):
    derived = cosmo.get_current_derived_parameters(['A_s', 'n_s', 'alpha_s'])
    A_s     = derived['A_s']
    n_s     = derived['n_s']
    alpha_s = derived.get('alpha_s', 0.0)
    pivot   = 0.05

    def find_pk_kmax(cosmo, z, k_low=1e-6, k_high=100.0, n_iter=60):
        # Ensure k_low works
        cosmo.pk(k_low, z)

        # Ensure k_high fails; if not, expand until it fails or cap
        kh = k_high
        for _ in range(12):
            try:
                cosmo.pk(kh, z)
                k_low = kh
                kh *= 2.0
            except Exception:
                k_high = kh
                break
        else:
            return k_low  # never failed (unlikely)

        lo, hi = k_low, k_high
        for _ in range(n_iter):
            mid = 0.5 * (lo + hi)
            try:
                cosmo.pk(mid, z)
                lo = mid
            except Exception:
                hi = mid
        return lo


    def primordial_spectrum(k):
        k = np.asarray(k)
        log_k_ratio = np.log(k / pivot)
        return (np.pi * np.sqrt(2 * A_s) * k**(-1.5) *
                (k / pivot)**((n_s - 1) / 2) *
                np.exp((alpha_s / 4.0) * log_k_ratio**2))

    # Transfers
    tf = cosmo.get_transfer(z)
    k_tf_h = tf['k (h/Mpc)']
    k_tf   = k_tf_h * cosmo.h()

    T_b   = tf['d_b']
    T_cdm = tf['d_cdm']
    T_idm = tf.get('d_idm_drmd', 0.0)
    T_nu = tf.get('d_ncdm[0]', 0.0)
    T_m_CLASS = tf['d_m']

    # Background at this redshift
    bg = cosmo.get_background()
    idx = np.argmin(np.abs(bg['z'] - z))

    rho_b   = bg['(.)rho_b'][idx]
    rho_cdm = bg['(.)rho_cdm'][idx]
    rho_idm = bg.get('(.)rho_idm_drmd', np.zeros_like(bg['z']))[idx]
    rho_nu = bg.get('(.)rho_ncdm[0]', np.zeros_like(bg['z']))[idx]

    rho_m = rho_b + rho_cdm + rho_idm + rho_nu
    w_b   = rho_b   / rho_m
    w_cdm = rho_cdm / rho_m
    w_idm = rho_idm / rho_m
    w_nu = rho_nu / rho_m

    T_cb = (w_b * T_b + w_cdm * T_cdm) / (w_b + w_cdm)
    T_m = w_b * T_b + w_cdm * T_cdm + w_idm * T_idm + w_nu * T_nu
    P_pri = primordial_spectrum(k_tf)

    # --------------------------
    # NONLINEAR PK WITH RETRIES
    # --------------------------
    if nonlinear:
        k_tf_h = np.asarray(k_tf_h)

        # Find pk kmax and mask to it
        k_h_max = find_pk_kmax(cosmo, z, k_low=1e-6, k_high=np.max(k_tf_h), n_iter=60)
        k_h_max *= (1 - 1e-10)  # tiny safety margin

        mask = k_tf_h <= k_h_max
        k_for_pk_h = k_tf_h[mask] * cosmo.h()

        P_k = np.array([cosmo.pk(kh, z) for kh in k_for_pk_h])

        # ALSO mask the returned k grid to match P_k length
        k_tf = k_tf[mask]
        P_pri = P_pri[mask]
        T_m = T_m[mask]
        T_cb = T_cb[mask]
        T_b = T_b[mask]
        T_cdm = T_cdm[mask]
        if np.ndim(T_idm) > 0:
            T_idm = T_idm[mask]
    else:
        P_k = (P_pri * T_m)**2



    return {
        'k'     : k_tf,
        'P_k'   : P_k,
        'P_pri' : P_pri,
        'w_b'   : w_b,
        'w_cdm' : w_cdm,
        'w_idm' : w_idm,
        'w_nu'  : w_nu,
        'T_m'   : T_m,
        'T_cb'  : T_cb,
        'T_b'   : T_b,
        'T_cdm' : T_cdm,
        'T_idm' : T_idm,
        'T_nu'  : T_nu,
        'T_m_CLASS': T_m_CLASS
    }