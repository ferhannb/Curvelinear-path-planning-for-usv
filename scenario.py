import numpy as np

from .mpc import MPCNumericClothoidCost


def default_parameters():
    return {
        "N_MPC": 20,
        "w_pos": 75.0,
        "w_psi": 200.0,
        "w_K": 5.0,
        "w_Kcmd": 0.25,
        "w_dKcmd": 15.0,
        "w_ds_smooth": 0.1,
        "w_Kf": 150.0,
        "w_prog": 0.0,
        "alpha_prog": 0.0,
        "hit_ratio": 0.7,
        "w_wp_intermediate": 5.0,
        "w_wp_final": 1.0,
        "term_scale_intermediate": 0.2,
        "term_scale_final": 1.0,
        "hit_scale_intermediate": 0.7,
        "ds_max": 3.0,
        "ds_jump_max": 0.0,
        "K_max": 0.3,
        "blocks_kcmd": [1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        "blocks_ds": [1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        "x0": 0.0,
        "y0": 0.0,
        "psi0": 0.0,
        "K0": 0.0,
        "waypoints": [
            {"X": 30.0, "Y": 0.0, "tol": 0.5, "psig": 0.0, "Kf": -0.0, "use_Kf": True},
            {"X": 60.0, "Y": 40.0, "tol": 0.5, "psig": 1, "Kf": 0.0, "use_Kf": True},
            {"X": 30.0, "Y": 60.0, "tol": 0.5, "psig": np.pi/3*2, "Kf": 0.0, "use_Kf": True},
            {"X": -30.0, "Y": 60.0, "tol": 0.5, "psig": np.pi, "Kf": 0.0, "use_Kf": True},
        ],
        "use_wp_kf": True,
        "kf_fallback": 0.0,
    }


def build_default_mpc(cfg=None):
    p = default_parameters()
    if cfg:
        p.update(cfg)

    mpc = MPCNumericClothoidCost(
        N=p["N_MPC"],
        w_pos=p["w_pos"],
        w_psi=p["w_psi"],
        w_K=p["w_K"],
        w_Kcmd=p["w_Kcmd"],
        w_dKcmd=p["w_dKcmd"],
        w_ds_smooth=p["w_ds_smooth"],
        ds_jump_max=p["ds_jump_max"],
        w_Kf=p["w_Kf"],
        ds_max=p["ds_max"],
        K_MAX=p["K_max"],
        block_lengths_Kcmd=p["blocks_kcmd"],
        block_lengths_ds=p["blocks_ds"],
        w_prog=p["w_prog"],
        alpha_prog=p["alpha_prog"],
        hit_ratio=p["hit_ratio"],
    )
    return mpc, p


def run_default(plot=True, cfg=None):
    mpc, p = build_default_mpc(cfg=cfg)

    log = mpc.run_receding_horizon_multi(
        waypoints=p["waypoints"],
        x0=p["x0"],
        y0=p["y0"],
        psi0=p["psi0"],
        K0=p["K0"],
        use_heading_gate=True,
        tol_psi=np.deg2rad(12.0),
        w_wp_intermediate=p["w_wp_intermediate"],
        term_scale_intermediate=p["term_scale_intermediate"],
        term_scale_final=p["term_scale_final"],
        hit_scale_intermediate=p["hit_scale_intermediate"],
        w_wp_final=p["w_wp_final"],
        use_wp_kf=p["use_wp_kf"],
        kf_fallback=p["kf_fallback"],
    )

    if plot:
        mpc.plot_receding(log)

    return mpc, log, p
