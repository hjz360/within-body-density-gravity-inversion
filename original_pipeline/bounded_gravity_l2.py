"""Nonnegative depth-weighted L2 reference; not a GRAV3D reproduction.

Solve in q = W m:
  .5 ||(A W^-1 / s) q - d/s||^2
  + .5 ratio^2 (||q||^2 + ell^2 ||D q||^2),  q >= 0.
D contains nearest-neighbour forward differences on all three grid axes.
No differences are imposed across the exterior boundary. ell is in cells.
The matrix norm bound used in s and the graph Laplacian bound ensure a safe
fixed step. Projection is part of each optimization step, not postprocessing.
"""
from __future__ import annotations

import time
import numpy as np
import torch


def depth_weights(shape, cell_m=50., z0_m=50., beta=2.):
    """Cell-centre approximation to normalized (z+z0)^(-beta/2)."""
    if len(shape) != 3 or min(shape) < 1 or cell_m <= 0 or z0_m <= 0 or beta < 0:
        raise ValueError("invalid grid/depth-weight parameters")
    z = (np.arange(shape[0], dtype=np.float64) + .5) * cell_m
    w = ((z + z0_m) / (z[0] + z0_m)) ** (-beta / 2.)
    return np.broadcast_to(w[:, None, None], shape).copy().reshape(-1)


def laplacian(q, shape):
    """D.T D q, batch first, flattened in z,y,x order."""
    x = q.reshape((-1,) + tuple(shape))
    out = torch.zeros_like(x)
    for axis in (1, 2, 3):
        left, right = [slice(None)] * 4, [slice(None)] * 4
        left[axis], right[axis] = slice(None, -1), slice(1, None)
        left, right = tuple(left), tuple(right)
        delta = x[right] - x[left]
        out[left] -= delta
        out[right] += delta
    return out.reshape(q.shape)


def objective_and_gradient(b, target, q, rho, shape, ell=1.):
    residual = q @ b.T - target
    rq = q + ell**2 * laplacian(q, shape)
    objective = .5 * (residual * residual).sum(1) + .5 * rho * (q * rq).sum(1)
    gradient = residual @ b + rho[:, None] * rq
    return objective, gradient


def optimality(b, target, q, rho, shape, ell=1.):
    """Float64 projected-gradient check and conservative strong-convexity gap."""
    objective, grad = objective_and_gradient(b, target, q, rho, shape, ell)
    lipschitz = 1. + rho * (1. + 12. * ell**2)
    pg = lipschitz[:, None] * (q - torch.clamp(q - grad/lipschitz[:, None], min=0.))
    scale = torch.clamp((target @ b).abs().amax(1), min=1e-30)
    relative_pg = pg.abs().amax(1) / scale
    # Strong convexity mu = rho; minimize the quadratic lower bound over q>=0.
    delta = torch.clamp(q - grad / rho[:, None], min=0.) - q
    gap = torch.clamp(-(grad*delta).sum(1) - .5*rho*(delta*delta).sum(1), min=0.)
    return objective, relative_pg, gap


class BoundedL2:
    def __init__(self, matrix, shape, *, weights=None, device="cpu", dtype=torch.float32, ell=1.):
        matrix = np.asarray(matrix, dtype=np.float64)
        self.shape, self.ell = tuple(shape), float(ell)
        if matrix.ndim != 2 or matrix.shape[1] != int(np.prod(shape)):
            raise ValueError("matrix/grid shape mismatch")
        if not np.isfinite(matrix).all() or ell < 0:
            raise ValueError("nonfinite matrix or negative smoothness length")
        w = np.ones(matrix.shape[1]) if weights is None else np.asarray(weights, dtype=np.float64)
        if w.shape != (matrix.shape[1],) or not np.isfinite(w).all() or np.any(w <= 0):
            raise ValueError("weights must be finite positive cell weights")
        bw = matrix / w[None, :]
        # ||B||_2^2 <= ||B||_1 ||B||_infinity, including signed matrices.
        self.scale = float(np.sqrt(np.abs(bw).sum(0).max()*np.abs(bw).sum(1).max()))
        if self.scale <= 0:
            raise ValueError("zero forward operator")
        self.b = torch.as_tensor(bw/self.scale, dtype=dtype, device=device)
        self.w = torch.as_tensor(w, dtype=dtype, device=device)
        self.b64 = torch.as_tensor(bw/self.scale, dtype=torch.float64, device=device)
        self.w64 = torch.as_tensor(w, dtype=torch.float64, device=device)

    @torch.no_grad()
    def solve(self, observations, ratios, *, max_iter=12000, tolerance=1e-6,
              check_every=100, initial_q=None, progress=None, gap_tolerance=None):
        data = np.asarray(observations, dtype=np.float64)
        if data.ndim != 2 or data.shape[1] != self.b.shape[0] or not np.isfinite(data).all():
            raise ValueError("observations must be finite (batch,station) arrays")
        ratios = np.broadcast_to(np.asarray(ratios, dtype=np.float64), (len(data),)).copy()
        if np.any(~np.isfinite(ratios)) or np.any(ratios <= 0):
            raise ValueError("ratios must be positive")
        if max_iter < 1 or check_every < 1 or tolerance <= 0:
            raise ValueError("invalid stopping settings")
        if gap_tolerance is not None and gap_tolerance <= 0:
            raise ValueError("gap tolerance must be positive")
        target = torch.as_tensor(data/self.scale, device=self.b.device, dtype=self.b.dtype)
        target64 = torch.as_tensor(data/self.scale, device=self.b.device, dtype=torch.float64)
        rho = torch.as_tensor(ratios**2, dtype=self.b.dtype, device=self.b.device)
        rho64 = torch.as_tensor(ratios**2, dtype=torch.float64, device=self.b.device)
        l = (1. + rho*(1. + 12.*self.ell**2))[:, None]
        q = torch.zeros((len(data), self.b.shape[1]), dtype=self.b.dtype, device=self.b.device)
        if initial_q is not None:
            q.copy_(torch.as_tensor(initial_q, dtype=q.dtype, device=q.device))
            if not torch.isfinite(q).all() or torch.any(q < 0):
                raise ValueError("initial iterate must be feasible")
        y, t = q.clone(), torch.ones((len(data), 1), dtype=q.dtype, device=q.device)
        start, history = time.perf_counter(), []
        converged = torch.zeros(len(data), dtype=torch.bool, device=q.device)
        for iteration in range(1, max_iter + 1):
            _, grad = objective_and_gradient(self.b, target, y, rho, self.shape, self.ell)
            updated = torch.clamp(y - grad/l, min=0.)
            tnew = (1. + torch.sqrt(1. + 4.*t*t)) / 2.
            # Adaptive gradient restart, independently per observation.
            restart = ((y-updated)*(updated-q)).sum(1, keepdim=True) > 0
            momentum = torch.where(restart, torch.zeros_like(t), (t-1.)/tnew)
            y = updated + momentum*(updated-q)
            q, t = updated, torch.where(restart, torch.ones_like(tnew), tnew)
            if iteration % check_every == 0 or iteration == max_iter:
                obj, pg, gap = optimality(self.b64, target64, q.double(), rho64, self.shape, self.ell)
                if not torch.isfinite(obj).all() or not torch.isfinite(pg).all():
                    raise FloatingPointError("nonfinite optimization result")
                converged = pg <= tolerance
                relative_gap = gap / torch.clamp(obj, min=1e-30)
                if gap_tolerance is not None:
                    converged &= relative_gap <= gap_tolerance
                history.append({"iteration": iteration, "elapsed_seconds": time.perf_counter()-start,
                                "objective": obj.cpu().tolist(), "relative_projected_gradient": pg.cpu().tolist(),
                                "relative_gap_upper_bound": relative_gap.cpu().tolist(),
                                "converged_count": int(converged.sum().item())})
                if progress is not None:
                    progress(history[-1])
                if bool(converged.all()):
                    break
        density = (q.double()/self.w64).cpu().numpy().reshape((-1,)+self.shape)
        # Last score corresponds exactly to returned q / w64. No clipping here.
        return density, {"iterations": iteration, "elapsed_seconds": time.perf_counter()-start,
                         "converged": converged.cpu().tolist(), "objective": obj.cpu().tolist(),
                         "relative_projected_gradient": pg.cpu().tolist(),
                         "strong_convexity_gap_upper_bound": gap.cpu().tolist(),
                         "relative_gap_upper_bound": relative_gap.cpu().tolist(),
                         "gap_tolerance": gap_tolerance,
                         "scale": self.scale, "tolerance": tolerance, "history": history,
                         "dtype": str(q.dtype), "verification_dtype": "torch.float64"}
