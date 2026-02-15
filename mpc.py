import time
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

from .helpers import (
    wrap_to_pi,
    clothoid_increment_numeric,
    K_next_fixed_ramp,
    wrap_to_pi_np,
    clothoid_increment_numeric_np,
    K_next_fixed_ramp_np,
)

class MPCNumericClothoidCost:
    def __init__(
        self,
        N=25,
        ds_min=0.01,
        ds_max=1.0,
        K_MAX=0.30,
        S_MAX=14.0,
        nseg=4,
        w_pos=50.0,
        w_psi=250.0,
        w_K=0.5,
        w_Kcmd=0.5,
        w_dKcmd=2.0,
        w_ds_smooth=1.0,
        ds_jump_max=None,   # hard bound on ds step change; None disables
        w_Kf=10.0,
        enable_terminal_K_hard=False,
        ipopt_max_iter=2000,
        ipopt_tol=1e-6,
        block_lengths_Kcmd=None,
        block_lengths_ds=None,
        # --- NEW: progress + tube cost
        w_prog=0.0,          # progress reward weight (objective uses -w_prog*sum(ds))
        alpha_prog=0.0,      # sum(ds) >= alpha_prog * Dref
        hit_ratio=0.7,       # waypoint hit index ratio inside horizon
    ):
        self.N = int(N)
        self.ds_min = float(ds_min)
        self.ds_max = float(ds_max)
        self.K_MAX = float(K_MAX)
        self.S_MAX = float(S_MAX)
        self.nseg = int(nseg)
        self.w_pos = float(w_pos)
        self.w_psi = float(w_psi)
        self.w_K = float(w_K)
        self.w_Kcmd = float(w_Kcmd)
        self.w_dKcmd = float(w_dKcmd)
        self.w_ds_smooth = float(w_ds_smooth)
        self.ds_jump_max = None if ds_jump_max is None else float(abs(ds_jump_max))
        self.w_Kf = float(w_Kf)
        self.enable_terminal_K_hard = bool(enable_terminal_K_hard)
        self.ipopt_max_iter = int(ipopt_max_iter)
        self.ipopt_tol = float(ipopt_tol)
        self.block_lengths_Kcmd = block_lengths_Kcmd
        self.block_lengths_ds = block_lengths_ds

        self.w_prog = float(w_prog)
        self.alpha_prog = float(alpha_prog)
        self.hit_ratio = float(hit_ratio)
        self.k_hit = int(round(self.hit_ratio * self.N))
        if self.k_hit < 1:
            self.k_hit = 1
        if self.k_hit > self.N - 1:
            self.k_hit = self.N - 1

        self._compute_block_maps()
        self._build_solver()
        self.last_sol = None
        self.last_warm = None
        self.last_ds_applied = float(self.ds_max)

    def _compute_block_maps(self):
        N = self.N

        def make_map(lengths):
            if lengths is None:
                return None, None
            if sum(lengths) != N:
                raise ValueError(
                    f"block_lengths toplamı N ile aynı olmalı (sum={sum(lengths)}, N={N})"
                )
            blk = np.empty(N, dtype=np.int32)
            s = 0
            for i, L in enumerate(lengths):
                blk[s : s + L] = i
                s += L
            return blk, len(lengths)

        self.bl_kcmd, self.NBK = make_map(self.block_lengths_Kcmd)
        self.bl_ds, self.NBd = make_map(self.block_lengths_ds)

    def _block_init_from_full(self, values, blk_map, n_blocks):
        out = np.empty(n_blocks, dtype=float)
        for i in range(n_blocks):
            sel = values[blk_map == i]
            out[i] = float(sel.mean()) if sel.size else float(values[0])
        return out

    def _build_solver(self):
        N = self.N
        opti = ca.Opti()

        X    = opti.variable(4, N + 1)  # x, y, psi, K
        Kcmd = opti.variable(1, N)
        ds   = opti.variable(1, N)

        x   = X[0, :]
        y   = X[1, :]
        psi = X[2, :]
        K   = X[3, :]

        # base parameters
        # initial state (current MPC state)
        x0_p        = opti.parameter()  # x0 [m]
        y0_p        = opti.parameter()  # y0 [m]
        psi0_p      = opti.parameter()  # psi0 [rad]
        K0_p        = opti.parameter()  # K0 [1/m]

        # terminal target (goal state at horizon end)
        xg_p        = opti.parameter()  # goal x [m]
        yg_p        = opti.parameter()  # goal y [m]
        psig_p      = opti.parameter()  # goal heading [rad]
        Kf_p        = opti.parameter()  # goal curvature [1/m]

        # interior hit target at k_hit (used to force passing near waypoint)
        xhit_p      = opti.parameter()  # hit x [m]
        yhit_p      = opti.parameter()  # hit y [m]
        psihit_p    = opti.parameter()  # hit heading [rad]
        Khit_p      = opti.parameter()  # hit curvature [1/m]
        hit_scale_p = opti.parameter()  # scale for interior hit cost terms

        # multi-waypoint cost shaping parameters
        ds_prev_p    = opti.parameter()  # applied ds from previous receding step
        term_scale_p = opti.parameter()  # scales terminal pos/psi terms (small for intermediate WP)
        w_wp_p       = opti.parameter()  # running tube-cost weight along horizon
        xwp_p        = opti.parameter()  # tube center x (active waypoint)
        ywp_p        = opti.parameter()  # tube center y (active waypoint)

        opti.subject_to(x[0] == x0_p)
        opti.subject_to(y[0] == y0_p)
        opti.subject_to(psi[0] == psi0_p)
        opti.subject_to(K[0] == K0_p)

        if self.bl_kcmd is not None:
            KcmdB = opti.variable(1, self.NBK)
            opti.subject_to(opti.bounded(-self.K_MAX, KcmdB, self.K_MAX))
            for k in range(N):
                opti.subject_to(Kcmd[0, k] == KcmdB[0, int(self.bl_kcmd[k])])
        else:
            KcmdB = None
            opti.subject_to(opti.bounded(-self.K_MAX, Kcmd, self.K_MAX))

        opti.subject_to(opti.bounded(-self.K_MAX, K, self.K_MAX))

        if self.bl_ds is not None:
            dsB = opti.variable(1, self.NBd)
            opti.subject_to(opti.bounded(self.ds_min, dsB, self.ds_max))
            for k in range(N):
                opti.subject_to(ds[0, k] == dsB[0, int(self.bl_ds[k])])
        else:
            dsB = None
            opti.subject_to(opti.bounded(self.ds_min, ds, self.ds_max))
        if self.ds_jump_max is not None and self.ds_jump_max > 0.0:
            for k in range(1, N):
                opti.subject_to(opti.bounded(-self.ds_jump_max, ds[k] - ds[k - 1], self.ds_jump_max))
            opti.subject_to(opti.bounded(-self.ds_jump_max, ds[0] - ds_prev_p, self.ds_jump_max))

        # dynamics
        for k in range(N):
            ds_k = ds[0, k]
            K1 = K_next_fixed_ramp(K[k], Kcmd[0, k], ds_k, K_MAX=self.K_MAX, S_MAX=self.S_MAX)
            x1, y1, psi1 = clothoid_increment_numeric(
                x[k], y[k], psi[k], K[k], K1, ds_k, nseg=self.nseg
            )
            opti.subject_to(K[k + 1] == K1)
            opti.subject_to(x[k + 1] == x1)
            opti.subject_to(y[k + 1] == y1)
            opti.subject_to(psi[k + 1] == psi1)

        #### Objective Function ###
        obj = 0
        for k in range(N):
            obj += self.w_K * (K[k] ** 2)
            obj += self.w_Kcmd * (Kcmd[0, k] ** 2)
            if k > 0:
                obj += self.w_dKcmd * (Kcmd[0, k] - Kcmd[0, k - 1]) ** 2
                obj += self.w_ds_smooth * (ds[k] - ds[k - 1]) ** 2

        epsD = 1e-6
        Dref = ca.sqrt((xg_p - x0_p) ** 2 + (yg_p - y0_p) ** 2)
        Dref_safe = Dref + 1e-3  # avoid divide-by-zero; also keeps constraint sane near zero

        # --- NEW: progress reward/constraint (disabled by default)
        # if self.w_prog > 0.0 or self.alpha_prog > 0.0:
        #     L = ca.sum2(ds)
        #     if self.w_prog > 0.0:
        #         obj += -self.w_prog * L
        #     if self.alpha_prog > 0.0:
        #         L_max = float(self.ds_max * self.N)
        #         L_target = ca.fmin(self.alpha_prog * Dref_safe, 0.95 * L_max)
        #         opti.subject_to(L >= L_target)

        # terminal costs (scaled)
        pos_e = (x[N] - xg_p) ** 2 + (y[N] - yg_p) ** 2
        psi_e = wrap_to_pi(psi[N] - psig_p)
        K_e = K[N] - Kf_p
        obj += term_scale_p * (self.w_pos * pos_e / (Dref_safe ** 2 + epsD))
        obj += term_scale_p * (self.w_psi * (psi_e ** 2))
        obj += term_scale_p * (self.w_Kf * (K_e ** 2))

        # --- NEW: tube cost for intermediate WP (sum along horizon)
        d2 = (x[1:] - xwp_p) ** 2 + (y[1:] - ywp_p) ** 2  # stages 1..N
        obj += w_wp_p * (ca.sum2(d2) / (Dref_safe ** 2 + epsD))

        # --- NEW: waypoint hit cost at a fixed interior index (prevents loitering)
        k_hit = self.k_hit
        pos_hit = (x[k_hit] - xhit_p) ** 2 + (y[k_hit] - yhit_p) ** 2
        psi_hit = wrap_to_pi(psi[k_hit] - psihit_p)
        obj += hit_scale_p * (self.w_pos * pos_hit / (Dref_safe ** 2 + epsD))
        obj += hit_scale_p * (self.w_psi * (psi_hit ** 2))
        obj += hit_scale_p * (self.w_Kf * (K[k_hit] - Khit_p) ** 2)

        # terminal curvature handling stays
        if self.enable_terminal_K_hard:
            opti.subject_to(K[N] == Kf_p)


        opti.minimize(obj)
        opti.solver(
            "ipopt",
            {"print_time": True},
            {
                "max_iter": self.ipopt_max_iter,
                "print_level": 1,
                "tol": self.ipopt_tol,
                "print_timing_statistics": "yes",
            },
        )

        self.opti = opti
        self.vars = {"X": X, "Kcmd": Kcmd, "ds": ds}
        if KcmdB is not None:
            self.vars["KcmdB"] = KcmdB
        if dsB is not None:
            self.vars["dsB"] = dsB
        self.states = {"x": x, "y": y, "psi": psi, "K": K}
        self.pars = {
            "x0": x0_p,
            "y0": y0_p,
            "psi0": psi0_p,
            "K0": K0_p,
            "xg": xg_p,
            "yg": yg_p,
            "psig": psig_p,
            "Kf": Kf_p,
            # new
            "ds_prev": ds_prev_p,
            "term_scale": term_scale_p,
            "w_wp": w_wp_p,
            "xwp": xwp_p,
            "ywp": ywp_p,
            "xhit": xhit_p,
            "yhit": yhit_p,
            "psihit": psihit_p,
            "Khit": Khit_p,
            "hit_scale": hit_scale_p,
        }

    def _set_params(
        self,
        x0,
        y0,
        psi0,
        K0,
        xg,
        yg,
        psig,
        Kf,
        term_scale,
        w_wp,
        xwp,
        ywp,
        xhit,
        yhit,
        psihit,
        Khit,
        hit_scale,
        ds_prev,
    ):
        self.opti.set_value(self.pars["x0"], x0)
        self.opti.set_value(self.pars["y0"], y0)
        self.opti.set_value(self.pars["psi0"], psi0)
        self.opti.set_value(self.pars["K0"], K0)
        self.opti.set_value(self.pars["xg"], xg)
        self.opti.set_value(self.pars["yg"], yg)
        self.opti.set_value(self.pars["psig"], psig)
        self.opti.set_value(self.pars["Kf"], Kf)
        self.opti.set_value(self.pars["xhit"], float(xhit))
        self.opti.set_value(self.pars["yhit"], float(yhit))
        self.opti.set_value(self.pars["psihit"], float(psihit))
        self.opti.set_value(self.pars["Khit"], float(Khit))
        self.opti.set_value(self.pars["hit_scale"], float(hit_scale))
        self.opti.set_value(self.pars["ds_prev"], float(ds_prev))
        self.opti.set_value(self.pars["term_scale"], float(term_scale))
        self.opti.set_value(self.pars["w_wp"], float(w_wp))
        self.opti.set_value(self.pars["xwp"], float(xwp))
        self.opti.set_value(self.pars["ywp"], float(ywp))

    def _warm_start(self, x0, y0, psi0, K0, xg, yg, ds_seed=None):
        """ According to actual states, the trajectories of states are calculated to provide an initial estimate for optimisation."""
        dist = np.hypot(xg - x0, yg - y0)
        psi_goal = np.arctan2(yg - y0, xg - x0)
        dpsi_goal = wrap_to_pi_np(psi_goal - psi0)
        Kcmd_guess = np.clip(dpsi_goal / max(dist, 1e-3), -self.K_MAX, self.K_MAX)
        Kcmd_ws = np.full(self.N, Kcmd_guess, dtype=float)

        # warm-start for ds:
        # - default heuristic from distance
        # - optionally seed with previous applied ds at waypoint transitions
        if ds_seed is None:
            ds_guess = np.clip(dist / max(self.N, 1), self.ds_min, self.ds_max)
            ds_guess = max(ds_guess, 0.6 * self.ds_max) if dist > 2.0 else ds_guess
        else:
            ds_guess = float(np.clip(ds_seed, self.ds_min, self.ds_max))
        ds_ws = np.full(self.N, ds_guess, dtype=float)

        x_ws = np.zeros(self.N + 1)
        y_ws = np.zeros(self.N + 1)
        psi_ws = np.zeros(self.N + 1)
        K_ws = np.zeros(self.N + 1)
        x_ws[0], y_ws[0], psi_ws[0], K_ws[0] = x0, y0, psi0, K0

        for k in range(self.N):
            K_ws[k + 1] = K_next_fixed_ramp_np(
                K_ws[k], Kcmd_ws[k], ds_ws[k], K_MAX=self.K_MAX, S_MAX=self.S_MAX
            )
            x_ws[k + 1], y_ws[k + 1], psi_ws[k + 1] = clothoid_increment_numeric_np(
                x_ws[k], y_ws[k], psi_ws[k], K_ws[k], K_ws[k + 1], ds_ws[k], nseg=self.nseg
            )

        self.opti.set_initial(self.vars["Kcmd"], Kcmd_ws.reshape(1, -1))
        self.opti.set_initial(self.vars["ds"], ds_ws.reshape(1, -1))
        if "KcmdB" in self.vars:
            kcmd_b = self._block_init_from_full(Kcmd_ws, self.bl_kcmd, self.NBK)
            self.opti.set_initial(self.vars["KcmdB"], kcmd_b.reshape(1, -1))
        if "dsB" in self.vars:
            ds_b = self._block_init_from_full(ds_ws, self.bl_ds, self.NBd)
            self.opti.set_initial(self.vars["dsB"], ds_b.reshape(1, -1))
        self.opti.set_initial(self.states["x"], x_ws)
        self.opti.set_initial(self.states["y"], y_ws)
        self.opti.set_initial(self.states["psi"], psi_ws)
        self.opti.set_initial(self.states["K"], K_ws)

    def _apply_warm_start(self, x0, y0, psi0, K0, xg, yg, use_last_warm=True, ds_seed=None):
        if use_last_warm and self.last_warm is not None:
            self.opti.set_initial(self.vars["X"], self.last_warm["X"])
            self.opti.set_initial(self.vars["Kcmd"], self.last_warm["Kcmd"].reshape(1, -1))
            self.opti.set_initial(self.vars["ds"], self.last_warm["ds"].reshape(1, -1))
            if "KcmdB" in self.vars:
                kcmd_b = self._block_init_from_full(self.last_warm["Kcmd"], self.bl_kcmd, self.NBK)
                self.opti.set_initial(self.vars["KcmdB"], kcmd_b.reshape(1, -1))
            if "dsB" in self.vars:
                ds_b = self._block_init_from_full(self.last_warm["ds"], self.bl_ds, self.NBd)
                self.opti.set_initial(self.vars["dsB"], ds_b.reshape(1, -1))
            return
        self._warm_start(x0, y0, psi0, K0, xg, yg, ds_seed=ds_seed)

    def _shift_solution(self, sol):
        X = sol["X"]
        Kcmd = sol["Kcmd"]
        ds = sol["ds"]

        X_shift = np.empty_like(X)
        X_shift[:, :-1] = X[:, 1:]
        X_shift[:, -1] = X[:, -1]

        Kcmd_shift = np.empty_like(Kcmd)
        Kcmd_shift[:-1] = Kcmd[1:]
        Kcmd_shift[-1] = Kcmd[-1]

        ds_shift = np.empty_like(ds)
        ds_shift[:-1] = ds[1:]
        ds_shift[-1] = ds[-1]

        self.last_warm = {"X": X_shift, "Kcmd": Kcmd_shift, "ds": ds_shift}

    def solve(
        self,
        x0=0.0,
        y0=0.0,
        psi0=0.0,
        K0=0.0,
        xg=45.0,
        yg=25.0,
        psig=0.0,
        Kf=0.0,
        # NEW:
        term_scale=1.0,
        w_wp=0.0,
        xwp=None,
        ywp=None,
        hit_scale=0.0,
        xhit=None,
        yhit=None,
        psihit=None,
        Khit=None,
        ds_prev=None,
        ds_seed=None,
        use_last_warm=True,
    ):
        if xwp is None:
            xwp = xg
        if ywp is None:
            ywp = yg
        if xhit is None:
            xhit = xg
        if yhit is None:
            yhit = yg
        if psihit is None:
            psihit = psig
        if Khit is None:
            Khit = Kf
        if ds_prev is None:
            ds_prev = self.last_ds_applied

        self._set_params(
            x0,
            y0,
            psi0,
            K0,
            xg,
            yg,
            psig,
            Kf,
            term_scale,
            w_wp,
            xwp,
            ywp,
            xhit,
            yhit,
            psihit,
            Khit,
            hit_scale,
            ds_prev,
        )
        self._apply_warm_start(
            x0, y0, psi0, K0, xg, yg, use_last_warm=use_last_warm, ds_seed=ds_seed
        )

        t0 = time.perf_counter()
        sol = self.opti.solve()
        t1 = time.perf_counter()

        Xsol = sol.value(self.vars["X"])
        Kcmdsol = sol.value(self.vars["Kcmd"]).ravel()
        dssol = sol.value(self.vars["ds"]).ravel()
        print(f"CasADi solve time: {t1 - t0:.3f} s")

        out = {
            "X": Xsol,
            "Kcmd": Kcmdsol,
            "ds": dssol,
            "start": (x0, y0, psi0, K0),
            "goal": (xg, yg, psig, Kf),
            "tube": (xwp, ywp, w_wp, term_scale),
        }
        self.last_sol = out
        return out

    def mpc_step(
        self,
        state,
        goal,
        term_scale=1.0,
        w_wp=0.0,
        xwp=None,
        ywp=None,
        hit_scale=0.0,
        xhit=None,
        yhit=None,
        psihit=None,
        Khit=None,
        ds_seed=None,
    ):
        x, y, psi, K = state
        xg, yg, psig, Kf = goal
        sol = self.solve(
            x, y, psi, K,
            xg, yg, psig, Kf,
            term_scale=term_scale,
            w_wp=w_wp,
            xwp=xwp, ywp=ywp,
            hit_scale=hit_scale,
            xhit=xhit, yhit=yhit, psihit=psihit, Khit=Khit,
            ds_prev=self.last_ds_applied,
            ds_seed=ds_seed,
            use_last_warm=True
        )

        ds0 = sol["ds"][0]
        K1 = sol["X"][3, 1]
        x1, y1, psi1 = clothoid_increment_numeric_np(
            x, y, psi, K, K1, ds0, nseg=self.nseg
        )
        self.last_ds_applied = float(ds0)

        self._shift_solution(sol)
        return (x1, y1, psi1, K1), (ds0, sol["Kcmd"][0]), sol

    def run_receding_horizon_multi(
        self,
        waypoints,
        x0=0.0,
        y0=0.0,
        psi0=0.0,
        K0=0.0,
        tol_default=2.5,
        max_iters=300,
        use_heading_gate=False,
        tol_psi=5 * np.pi / 180,
        w_wp_intermediate=0.0,   # tube cost weight for intermediate WP
        term_scale_intermediate=0.1,  # small terminal pos/psi at intermediate WP
        term_scale_final=1.0,     # keep terminal pos/psi at final WP
        hit_scale_intermediate=1.0,   # enforce hit at interior index for intermediate WP
        w_wp_final=2.0,          # running distance cost for final WP
        use_wp_kf=True,          # global flag: use waypoint Kf targets or ignore them
        kf_fallback=0.0,         # Kf value when WP Kf is disabled/missing
    ):
        if not waypoints:
            raise ValueError("waypoints listesi bos olamaz.")

        def pick_heading(x, y, Xf, Yf, psig_opt):
            if psig_opt is None:
                return np.arctan2(Yf - y, Xf - x)
            return float(psig_opt)

        def pick_Kf(wp):
            # Global flag + optional per-WP override ("use_Kf": True/False).
            if (not use_wp_kf) or (not bool(wp.get("use_Kf", True))):
                return float(kf_fallback)
            Kf_opt = wp.get("Kf", None)
            return float(kf_fallback) if (Kf_opt is None) else float(Kf_opt)

        self.last_warm = None
        self.last_ds_applied = float(self.ds_max)
        state = (x0, y0, psi0, K0)
        x, y, psi, K = state

        traj = [(x0, y0)]
        psi_hist = [psi0]
        K_hist = [K0]
        Kcmd_hist = []
        ds_hist = []
        idx_hist = [0]
        ds_seed_next = None

        cur_idx = 0

        for _ in range(max_iters):
            wp = waypoints[cur_idx]
            Xf, Yf = float(wp["X"]), float(wp["Y"])
            psig = pick_heading(x, y, Xf, Yf, wp.get("psig", wp.get("thf", None)))
            Kf = pick_Kf(wp)

            tol_here = float(wp.get("tol", tol_default))
            is_last = (cur_idx >= len(waypoints) - 1)

            # --- SET-BASED WP TRANSITION (disk membership)
            dist_now = np.hypot(x - Xf, y - Yf)
            heading_ok = True
            if use_heading_gate:
                heading_ok = abs(wrap_to_pi_np(psi - psig)) < tol_psi

            if (not is_last) and (dist_now <= tol_here) and heading_ok:
                # Force a fresh warm-start when switching active waypoint,
                # but keep ds continuity by seeding with the last applied step.
                self.last_warm = None
                ds_seed_next = ds_hist[-1] if ds_hist else None
                cur_idx += 1
                continue

            # --- COST SETUP
            if is_last:
                term_scale = term_scale_final
                w_wp = float(w_wp_final)
                hit_scale = 0.0
            else:
                term_scale = term_scale_intermediate
                w_wp = float(wp.get("w_wp", w_wp_intermediate))
                hit_scale = float(wp.get("hit_scale", hit_scale_intermediate))

            # solve step
            state, (ds0, Kcmd0), _ = self.mpc_step(
                state, (Xf, Yf, psig, Kf),
                term_scale=term_scale,
                w_wp=w_wp,
                xwp=Xf, ywp=Yf,
                hit_scale=hit_scale,
                xhit=Xf, yhit=Yf, psihit=psig, Khit=Kf,
                ds_seed=ds_seed_next,
            )
            ds_seed_next = None

            x, y, psi, K = state
            traj.append((x, y))
            psi_hist.append(psi)
            K_hist.append(K)
            Kcmd_hist.append(Kcmd0)
            ds_hist.append(ds0)
            idx_hist.append(cur_idx)

            # final stop condition
            if is_last:
                dist_now = np.hypot(x - Xf, y - Yf)
                heading_ok = True
                if use_heading_gate:
                    heading_ok = abs(wrap_to_pi_np(psi - psig)) < tol_psi
                if dist_now <= tol_here and heading_ok:
                    break

        wp_last = waypoints[min(cur_idx, len(waypoints) - 1)]
        Xf_last = float(wp_last["X"])
        Yf_last = float(wp_last["Y"])
        psig_last = pick_heading(x, y, Xf_last, Yf_last, wp_last.get("psig", wp_last.get("thf", None)))
        Kf_last = pick_Kf(wp_last)

        return {
            "traj": np.array(traj),
            "psi": np.array(psi_hist),
            "K": np.array(K_hist),
            "Kcmd": np.array(Kcmd_hist),
            "ds": np.array(ds_hist),
            "start": (x0, y0, psi0, K0),
            "goal": (Xf_last, Yf_last, psig_last, Kf_last),
            "waypoints": waypoints,
            "wp_index": np.array(idx_hist),
            "active_wp": cur_idx,
        }

    def plot_receding(self, log):
        x0, y0, _, _ = log["start"]
        xg, yg, _, _ = log["goal"]
        traj = log["traj"]
        psi_arr = log["psi"]
        K_arr = log["K"]
        Kcmd_arr = log["Kcmd"]
        ds_arr = log["ds"]

        s_grid = np.concatenate(([0.0], np.cumsum(ds_arr)))
        s_cmd = s_grid[:-1]
        idx_arr = np.asarray(log.get("wp_index", []), dtype=int)
        wp_start_state = {}
        wp_start_step = {}
        if ("waypoints" in log) and (idx_arr.size > 0):
            n_wp = len(log["waypoints"])
            idx_steps = idx_arr[1:] if idx_arr.size > 1 else np.array([], dtype=int)
            for i in range(n_wp):
                hits_state = np.where(idx_arr == i)[0]
                if hits_state.size > 0:
                    wp_start_state[i] = int(hits_state[0])
                hits_step = np.where(idx_steps == i)[0]
                if hits_step.size > 0:
                    wp_start_step[i] = int(hits_step[0])

        fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)

        ax = axes[0, 0]
        ax.plot(traj[:, 0], traj[:, 1], marker="o")
        ax.scatter([x0, xg], [y0, yg], marker="x")
        if "waypoints" in log:
            waypoints = log["waypoints"]
            WXs = [wp["X"] for wp in waypoints]
            WYs = [wp["Y"] for wp in waypoints]
            ax.scatter(WXs, WYs, marker="*", s=120)
            if idx_arr.size > 0:
                for i in range(len(waypoints)):
                    seg = (idx_arr == i)
                    if seg.any():
                        seg_traj = traj[seg]
                        n_pts = int(np.count_nonzero(seg))
                        ax.plot(seg_traj[:, 0], seg_traj[:, 1], linewidth=2, alpha=0.5)
                        mid = seg_traj[n_pts // 2]
                        ax.annotate(
                            f"WP{i} n={n_pts}",
                            (mid[0], mid[1]),
                            xytext=(6, 6),
                            textcoords="offset points",
                            fontsize=8,
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8),
                        )
                        if i in wp_start_state:
                            k0 = wp_start_state[i]
                            ax.annotate(
                                f"WP{i}",
                                (traj[k0, 0], traj[k0, 1]),
                                xytext=(-14, -12),
                                textcoords="offset points",
                                fontsize=9,
                                fontweight="bold",
                                bbox=dict(boxstyle="round,pad=0.2", fc="#fff3bf", ec="gray", alpha=0.95),
                            )
        ax.axis("equal")
        ax.grid(True)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title("XY trajectory (Receding Horizon)")

        ax = axes[0, 1]
        ax.plot(s_grid, K_arr, marker="o", label="K (state)")
        ax.plot(s_cmd, Kcmd_arr, "x--", label="K_cmd")
        for i, k0 in wp_start_state.items():
            if 0 <= k0 < len(K_arr):
                ax.annotate(
                    f"WP{i}",
                    (s_grid[k0], K_arr[k0]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                    fontweight="bold",
                )
        ax.grid(True)
        ax.legend()
        ax.set_xlabel("s [m]")
        ax.set_ylabel("K [1/m]")
        ax.set_title("Curvature profile")

        ax = axes[1, 0]
        ax.plot(s_grid, psi_arr, marker="o")
        for i, k0 in wp_start_state.items():
            if 0 <= k0 < len(psi_arr):
                ax.annotate(
                    f"WP{i}",
                    (s_grid[k0], psi_arr[k0]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                    fontweight="bold",
                )
        ax.grid(True)
        ax.set_xlabel("s [m]")
        ax.set_ylabel("psi [rad]")
        ax.set_title("Heading vs s")

        ax = axes[1, 1]
        ax.step(s_cmd, ds_arr, where="post")
        for i, k0 in wp_start_step.items():
            if 0 <= k0 < len(ds_arr):
                ax.annotate(
                    f"WP{i}",
                    (s_cmd[k0], ds_arr[k0]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                    fontweight="bold",
                )
        ax.grid(True)
        ax.set_xlabel("s [m]")
        ax.set_ylabel("ds [m]")
        ax.set_title("Step length profile")

        plt.show()

        print(
            "Terminal [x,y,psi,K] =",
            np.array([traj[-1, 0], traj[-1, 1], psi_arr[-1], K_arr[-1]]),
        )


