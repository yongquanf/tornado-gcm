# Copyright 2026 yongquan fu
# SPDX-License-Identifier: Apache-2.0
"""INV-5: declared precision must equal executed precision -- checked, not assumed.

INV-1..INV-4 check what a policy *declares*.  This module checks what a workload
actually *executes*, at dispatch level, and reports the difference.  It exists
because three separate defects of exactly this class were found in this repo:

  * a zone-level ``Z3=BF16`` declaration never reached the NN weights (the
    ``bake`` backend only cast modules listed in the per-module map);
  * the ``TF32`` flag was never asserted, so declared TF32 ran as FP32;
  * ``sht_adaptive_threshold`` was never consumed, so the spectral transform ran
    FP64 for every policy (48.2 GFLOP/step, the largest single component).

Usage::

    report = audit_declared_vs_executed(model, policy, steps=1)
    assert not report.mismatches, report.mismatches
"""
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import torch

# Model methods that delimit a zone; everything else is attributed to Z1/Z0.
_ZONE_ATTRS = (("Z3", "_apply_parameterization"), ("Z2", "_apply_conservation_fix"))
_DEFAULT_ZONE = "Z1/Z0"


@dataclass
class AuditReport:
    """Result of one declared-vs-executed audit."""

    declared: Dict[str, List[str]] = field(default_factory=dict)
    gflop: Dict[str, Dict[str, float]] = field(default_factory=dict)
    ops: Dict[str, Dict[str, int]] = field(default_factory=dict)
    mismatches: List[Dict[str, Any]] = field(default_factory=list)
    overhead_ms: Optional[float] = None
    audited_ms: Optional[float] = None
    plain_ms: Optional[float] = None

    @property
    def overhead_ratio(self) -> Optional[float]:
        if self.overhead_ms is None or self.plain_ms in (None, 0.0):
            return None
        return self.audited_ms / self.plain_ms

    def amortized_overhead_frac(self, rollout_steps: int) -> Optional[float]:
        """Fraction of total rollout time that one audited step adds."""
        if self.overhead_ratio is None or rollout_steps <= 0:
            return None
        return (self.overhead_ratio - 1.0) / rollout_steps

    def summary(self) -> str:
        lines = ["zone|dtype -> GFLOP"]
        for zone, bydt in sorted(self.gflop.items()):
            for dt, g in sorted(bydt.items(), key=lambda kv: -kv[1]):
                lines.append("  %-6s %-9s %8.2f" % (zone, dt, g))
        if self.mismatches:
            lines.append("mismatches:")
            for m in self.mismatches:
                lines.append("  %s executed %s but declared %s (%.2f GFLOP, %.1f%% of zone)"
                             % (m["zone"], m["executed"], "/".join(m["declared"]),
                                m["gflop"], 100.0 * m["share"]))
        else:
            lines.append("mismatches: none")
        return "\n".join(lines)


def _declared_sets(policy: Any) -> Dict[str, List[str]]:
    """Declared dtype names per zone, as strings like 'float32'."""
    def nm(dt):
        return str(dt).split(".")[-1] if dt is not None else None

    z0, z1 = nm(getattr(policy, "z0_dtype", None)), nm(getattr(policy, "z1_compute_dtype", None))
    z2 = nm(getattr(policy, "z2_compute_dtype", None))
    z3 = nm(getattr(policy, "z3_compute_dtype", None))
    # Z0 and Z1 share one dispatch bucket (both are matmul-heavy), but Z0 is
    # FP64 *by definition* while Z1 is the tunable one.  Compare the bucket
    # against the Z1 declaration: that is what catches the transform defect
    # (48.2 GFLOP of FP64 under a policy declaring Z1=FP32).  Z0's legitimate
    # FP64 work is O(quadrature), i.e. negligible in the GFLOP share -- the
    # reported share is what tells the two apart.
    out = {"Z1/Z0": [z1] if z1 else ([z0] if z0 else []),
           "Z2": [z2] if z2 else [], "Z3": [z3] if z3 else []}
    mdm = getattr(policy, "_module_dtype_map", None) or {}
    for v in mdm.values():
        n = nm(v)
        if n and n not in out["Z3"]:
            out["Z3"].append(n)
    return out


def audit_declared_vs_executed(
    model: torch.nn.Module,
    policy: Any = None,
    step_fn: Optional[Callable[..., Any]] = None,
    step_args: tuple = (),
    steps: int = 1,
    measure_overhead: bool = False,
) -> AuditReport:
    """Record the executed dtype of every dispatched op for ``steps`` calls.

    Args:
        model: the workload (used for zone attribution if it has the zone methods).
        policy: precision policy to compare against; taken from
            ``model.policy`` when omitted.  A policy is *required* -- without a
            declaration there is nothing to check (raises RuntimeError).
        step_fn: callable to drive one iteration; defaults to ``model.step``.
        step_args: positional args for ``step_fn``.
        steps: how many iterations to record.
        measure_overhead: also time one plain vs one audited iteration.

    Returns:
        AuditReport with the per-zone/dtype GFLOP table and the mismatch list.
    """
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils.flop_counter import flop_registry

    if policy is None:
        policy = getattr(model, "policy", None)
    if policy is None:
        raise RuntimeError(
            "INV-5 audit needs a policy to compare against: pass policy=... or "
            "set model.policy"
        )
    if step_fn is None:
        step_fn = getattr(model, "step", None)
    if step_fn is None:
        raise RuntimeError("no step_fn given and model has no .step()")

    zone = [_DEFAULT_ZONE]
    wrapped = []
    for zname, attr in _ZONE_ATTRS:
        fn = getattr(model, attr, None)
        if fn is None or not callable(fn):
            continue

        def _wrap(orig, z):
            def w(*args, **kw):
                prev = zone[0]
                zone[0] = z
                try:
                    return orig(*args, **kw)
                finally:
                    zone[0] = prev
            return w

        setattr(model, attr, _wrap(fn, zname))
        wrapped.append((attr, fn))

    flops: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    ops: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    class _Mode(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            out = func(*args, **kwargs)
            ts = [x for x in args if isinstance(x, torch.Tensor)]
            dt = str(ts[0].dtype).split(".")[-1] if ts else "-"
            ops[zone[0]][dt] += 1
            pk = func._overloadpacket
            if pk in flop_registry:
                try:
                    flops[zone[0]][dt] += flop_registry[pk](*args, **kwargs, out_val=out)
                except Exception:
                    pass
            return out

    report = AuditReport(declared=_declared_sets(policy))
    try:
        with torch.no_grad():
            with _Mode():
                for _ in range(max(1, int(steps))):
                    step_fn(*step_args)
    finally:
        for attr, orig in wrapped:
            setattr(model, attr, orig)

    report.gflop = {z: dict(v) for z, v in flops.items()}
    report.ops = {z: dict(v) for z, v in ops.items()}

    # Compare, weighting by matmul GFLOP (op counts alone over-weight elementwise ops).
    for z, bydt in report.gflop.items():
        allowed = report.declared.get(z, [])
        if not allowed:
            continue
        total = sum(bydt.values()) or 1.0
        for dt, g in bydt.items():
            if g <= 0 or dt in allowed or dt == "-":
                continue
            entry = {
                "zone": z, "executed": dt, "declared": list(allowed),
                "gflop": g, "share": g / total,
                "kind": "higher_than_declared" if dt == "float64" else "lower_than_declared",
            }
            if z == "Z1/Z0":
                entry["note"] = ("Z0 legitimately runs FP64; judge by the share -- "
                                 "a large share is the transform, not Z0")
            report.mismatches.append(entry)
    report.mismatches.sort(key=lambda m: -m["gflop"])

    if measure_overhead:
        def _time_one(fn, use_mode):
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t0 = time.perf_counter()
            with torch.no_grad():
                if use_mode:
                    with _Mode():
                        fn()
                else:
                    fn()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            return (time.perf_counter() - t0) * 1e3

        report.plain_ms = _time_one(lambda: step_fn(*step_args), False)
        report.audited_ms = _time_one(lambda: step_fn(*step_args), True)
        report.overhead_ms = report.audited_ms - report.plain_ms
    return report
