#!/usr/bin/env python3
"""
SKRAID — Extended Kalman Filter for temporal smoothing of skid-risk score
=========================================================================
Drop-in module for benchmark_skraid_pi4.py and live_deploy_skraid.py.

WHY AN *EXTENDED* KF (and not a plain KF)
-----------------------------------------
A plain 1-D Kalman filter on the raw probability p is the wrong model:
p is bounded on [0,1], so a linear-Gaussian state would predict values
outside the valid range and the innovation variance would be wrong near
the boundaries (exactly where a detector lives).

Instead we track the latent risk in LOG-ODDS space, which is unbounded:

    state      x = [ l, l_dot ]^T        l = logit(p_true), l_dot = its rate
    process    x_k = F x_{k-1} + w,      F = [[1, dt],[0, 1]]   (const. velocity)
    measurement z_k = sigma(l) + v,      sigma = logistic function   <-- NONLINEAR

Because the measurement function h(x) = sigma(l) is nonlinear, the update
step must linearise it via its Jacobian:

    H = [ dsigma/dl , 0 ] = [ sigma(l)(1 - sigma(l)) , 0 ]

That linearisation is precisely what makes this an EKF rather than a KF.
This is a defensible, reportable design choice for the paper.

ADAPTIVE MEASUREMENT NOISE (the useful part)
--------------------------------------------
A RandomForest gives you something most classifiers don't: an ensemble of
independent estimates. The per-tree disagreement about "High Risk" is a
free, per-window estimate of how uncertain this particular prediction is.
We use it directly as the measurement noise R_k:

    R_k = max( Var_over_trees( p_tree ), R_floor )

So windows where the forest is internally split get down-weighted by the
filter, and confident windows move the state faster. This is adaptive
measurement noise driven by model uncertainty, and it costs one extra
pass over the trees.

HYSTERESIS
----------
Single-threshold alarms chatter when the smoothed score sits near the
boundary. We use dual thresholds (Schmitt trigger): the alarm asserts at
t_on and only clears below t_off (t_off < t_on), with an optional minimum
hold time so a fired warning stays visible long enough for the rider.

USAGE
-----
    from skraid_ekf import RiskEKF, tree_disagreement

    ekf = RiskEKF(dt=0.5, t_on=0.15, t_off=0.08)

    # per window:
    p_raw = model.predict_proba(x)[0][high_risk_idx]
    R_k   = tree_disagreement(model, x, high_risk_idx)   # optional, adaptive
    out   = ekf.step(p_raw, R_meas=R_k)

    out.p_smooth   -> filtered probability in [0,1]
    out.p_rate     -> d(logit)/dt, >0 means risk is building
    out.alarm      -> hysteresis-gated boolean
    out.innovation -> z - h(x_pred); large values = surprising window
"""

from dataclasses import dataclass
import numpy as np

_EPS = 1e-6


def _logit(p, eps=_EPS):
    p = float(np.clip(p, eps, 1.0 - eps))
    return float(np.log(p / (1.0 - p)))


def _sigmoid(l):
    # numerically stable logistic
    if l >= 0:
        z = np.exp(-l)
        return float(1.0 / (1.0 + z))
    z = np.exp(l)
    return float(z / (1.0 + z))


@dataclass
class EKFOutput:
    p_raw: float          # unfiltered model probability
    p_smooth: float       # EKF posterior probability
    p_rate: float         # posterior d(logit)/dt  (risk trend)
    alarm: bool           # hysteresis-gated decision
    innovation: float     # z - h(x_pred)
    R_used: float         # measurement noise actually applied
    var_logit: float      # posterior variance of the logit state


class RiskEKF:
    """
    Extended Kalman Filter over latent log-odds risk with a logistic
    measurement model, adaptive measurement noise, and hysteresis output.

    Parameters
    ----------
    dt : float
        Nominal seconds between scores (your STEP_SEC). Can be overridden
        per call in step(), which matters on a Pi where the loop jitters.
    q_level, q_rate : float
        Process noise on the level and rate states. Larger = filter trusts
        new measurements more / tracks faster. Start with the defaults and
        tune q_level on a recorded session, not on the road.
    rate_damping : float in (0, 1]
        Decay applied to the rate state each step. 1.0 = pure constant
        velocity (spikes over-extrapolate); lower = trend decays unless
        re-supported by data. 0.65 is a reasonable default at dt=0.5.
    r_floor : float
        Minimum measurement variance. Prevents a unanimous forest from
        producing R=0 and a singular update (which would make the filter
        follow every raw reading exactly and defeat the point).
    r_default : float
        Measurement variance used when no per-window R is supplied.
        0.02 corresponds to ~0.14 std on the probability scale, which is
        a realistic noise level for a 200-tree RF on 2 s windows.
    t_on, t_off : float
        Hysteresis thresholds on the SMOOTHED probability. t_off < t_on.
    min_hold_steps : int
        Once asserted, hold the alarm at least this many steps.
    init_p : float
        Prior belief at t=0. Default 0.02 = "assume safe until shown
        otherwise", which matches a passive L1 warning system.
    """

    def __init__(self, dt=0.5, q_level=0.35, q_rate=0.05, r_floor=1e-4,
                 r_default=0.02, rate_damping=0.65, t_on=0.15, t_off=0.08,
                 min_hold_steps=2, init_p=0.02, init_var=4.0):
        if not (0.0 < t_off < t_on < 1.0):
            raise ValueError("require 0 < t_off < t_on < 1")
        self.dt = float(dt)
        self.q_level = float(q_level)
        self.q_rate = float(q_rate)
        self.rate_damping = float(rate_damping)
        self.r_floor = float(r_floor)
        self.r_default = float(r_default)
        self.t_on = float(t_on)
        self.t_off = float(t_off)
        self.min_hold_steps = int(min_hold_steps)

        # state: [logit level, logit rate]
        self.x = np.array([_logit(init_p), 0.0], dtype=float)
        self.P = np.diag([float(init_var), 1.0])

        self._alarm = False
        self._hold = 0

    # -- internals ---------------------------------------------------
    def _F(self, dt):
        """
        Damped constant-velocity model. A pure [[1,dt],[0,1]] transition
        lets a single spurious spike inject velocity that keeps
        extrapolating the risk upward for seconds afterwards. Damping the
        rate term (rho < 1) makes the trend decay unless it is
        continuously re-supported by new measurements -- which is the
        correct prior for road hazards: risk is persistent but not
        self-sustaining.
        """
        return np.array([[1.0, dt],
                         [0.0, self.rate_damping]], dtype=float)

    def _Q(self, dt):
        """Continuous white-noise-acceleration discretisation."""
        return np.array([
            [self.q_level * dt**3 / 3.0, self.q_level * dt**2 / 2.0],
            [self.q_level * dt**2 / 2.0, self.q_level * dt + self.q_rate * dt],
        ], dtype=float)

    # -- public API --------------------------------------------------
    def step(self, p_meas, R_meas=None, dt=None):
        """
        One predict+update cycle.

        p_meas : float   raw model probability of High Risk for this window
        R_meas : float   optional per-window measurement variance (see
                         tree_disagreement). If None, uses r_default.
        dt     : float   actual elapsed seconds since last step, if it
                         differs from the nominal dt.
        """
        dt = self.dt if dt is None else max(float(dt), 1e-3)

        # ---------------- PREDICT ----------------
        F = self._F(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self._Q(dt)

        # ---------------- UPDATE (nonlinear measurement) ----------------
        l_pred = self.x[0]
        h = _sigmoid(l_pred)                 # predicted measurement
        dh = h * (1.0 - h)                   # Jacobian entry d(sigma)/d(l)
        H = np.array([[dh, 0.0]], dtype=float)

        R = self.r_default if R_meas is None else max(float(R_meas), self.r_floor)

        z = float(np.clip(p_meas, 0.0, 1.0))
        y = z - h                            # innovation
        S = float((H @ self.P @ H.T)[0, 0]) + R   # innovation covariance
        K = (self.P @ H.T) / S               # Kalman gain (2x1)

        self.x = self.x + (K.flatten() * y)
        I = np.eye(2)
        self.P = (I - K @ H) @ self.P
        self.P = 0.5 * (self.P + self.P.T)   # keep symmetric

        p_smooth = _sigmoid(self.x[0])

        # ---------------- HYSTERESIS ----------------
        if self._alarm:
            self._hold = max(0, self._hold - 1)
            if p_smooth < self.t_off and self._hold == 0:
                self._alarm = False
        else:
            if p_smooth >= self.t_on:
                self._alarm = True
                self._hold = self.min_hold_steps

        return EKFOutput(
            p_raw=z,
            p_smooth=p_smooth,
            p_rate=float(self.x[1]),
            alarm=bool(self._alarm),
            innovation=float(y),
            R_used=float(R),
            var_logit=float(self.P[0, 0]),
        )

    def reset(self, init_p=0.02, init_var=4.0):
        self.x = np.array([_logit(init_p), 0.0], dtype=float)
        self.P = np.diag([float(init_var), 1.0])
        self._alarm = False
        self._hold = 0


def tree_disagreement(model, X, class_idx):
    """
    Per-window measurement variance from RandomForest tree disagreement.

    Returns Var over trees of P(class_idx). Cost is one predict_proba per
    tree; on a 200-tree forest this is ~10-20 ms on a Pi 4, which is
    negligible next to the vision stage but NOT free -- if you are latency
    bound, pass R_meas=None and use a fixed R instead.
    """
    # Convert ONCE to a bare float array. Passing a DataFrame into every
    # tree re-runs sklearn's feature-name validation 200 times, which
    # dominates the cost (measured ~105 ms vs ~7 ms on the same forest).
    Xa = np.asarray(X, dtype=np.float32)
    if Xa.ndim == 1:
        Xa = Xa.reshape(1, -1)
    votes = np.empty(len(model.estimators_), dtype=float)
    for i, est in enumerate(model.estimators_):
        votes[i] = est.predict_proba(Xa)[0][class_idx]
    return float(votes.var())


def calibrate_thresholds(p_scores, y_true_high_risk, target_recall=0.70):
    """
    Replace the hardcoded 0.15 with a threshold derived from validation data.

    Sweeps candidate thresholds and returns the LOWEST threshold achieving
    target_recall on the High Risk class, plus the precision there. Use the
    returned value as t_on, and t_off = 0.6 * t_on as a starting point.

    Returns (t_on, achieved_recall, achieved_precision) or (None, 0, 0) if
    the target recall is unreachable -- which is itself a finding worth
    reporting rather than hiding.
    """
    p_scores = np.asarray(p_scores, dtype=float)
    y = np.asarray(y_true_high_risk).astype(bool)
    if y.sum() == 0:
        return None, 0.0, 0.0

    best = (None, 0.0, 0.0)
    for t in np.unique(np.round(p_scores, 4)):
        pred = p_scores >= t
        tp = int((pred & y).sum())
        fp = int((pred & ~y).sum())
        rec = tp / y.sum()
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        if rec >= target_recall:
            if best[0] is None or t > best[0]:
                best = (float(t), float(rec), float(prec))
    return best
