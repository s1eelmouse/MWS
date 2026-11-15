"""Unified evidence recovery entry point.

NOISE_ROBUST_RECOVERY=1 runs the same rules on an ASR-repaired copy of the call and
snaps damaged tariff / ticket status values to the catalog (src/asr_repair.py).
"""
import os

from unified_recovery import recover_facts as _recover_facts, RECOVERY_POLICY as _BASE_POLICY

NOISE_ROBUST_RECOVERY = os.getenv('NOISE_ROBUST_RECOVERY', '0').strip().lower() in ('1', 'true', 'yes', 'on')
RECOVERY_POLICY = _BASE_POLICY + ('+asr-repair-v1' if NOISE_ROBUST_RECOVERY else '')


def recover_facts(facts, call, catalog_path=None, *, noise_robust=None):
    enabled = NOISE_ROBUST_RECOVERY if noise_robust is None else noise_robust
    if not enabled:
        return _recover_facts(facts, call, catalog_path)
    from asr_repair import repair_call, snap_values
    facts, snapped = snap_values(facts, catalog_path)
    output, logs = _recover_facts(facts, repair_call(call, catalog_path), catalog_path)
    for log in snapped + logs:
        log['policy'] = RECOVERY_POLICY if NOISE_ROBUST_RECOVERY else _BASE_POLICY + '+asr-repair-v1'
    return output, snapped + logs
