"""Rule engine, three-valued (BASELINE stand-in for the team's constraint matching).

status: "yes" (all conditions satisfied), "no" (some condition violated), "unknown" (nothing violated, but
a needed fact is missing). Missing means unknown, never false (Profile State Manager rule).
"""
from __future__ import annotations


def _is(v, want) -> bool:
    """Rule values may be a single value or a list of allowed values (e.g. occupation: ['farmer', 'entrepreneur'])."""
    return v in want if isinstance(want, (list, tuple, set)) else v == want


def check(rule: dict, p: dict) -> tuple[str, list[str], list[str]]:
    """Returns (status, unknown_fields, failed_fields) for one scheme rule against a profile dict."""
    unknown: list[str] = []
    failed: list[str] = []

    def need(field, ok):
        v = p.get(field)
        if v is None:
            if field not in unknown:
                unknown.append(field)
        elif not ok(v) and field not in failed:
            failed.append(field)

    if "state" in rule:      need("state", lambda v: _is(v, rule["state"]))
    if "occupation" in rule: need("occupation", lambda v: _is(v, rule["occupation"]))
    if "min_age" in rule:    need("age", lambda v: v >= rule["min_age"])
    if "max_age" in rule:    need("age", lambda v: v <= rule["max_age"])
    if "max_income" in rule: need("income", lambda v: v <= rule["max_income"])
    if "castes" in rule:     need("caste", lambda v: _is(v, rule["castes"]))
    if "gender" in rule:     need("gender", lambda v: _is(v, rule["gender"]))
    if rule.get("disability"): need("disability", lambda v: v is True)
    if rule.get("bpl"):        need("bpl", lambda v: v is True)
    if "residence" in rule:  need("residence", lambda v: _is(v, rule["residence"]))
    status = "no" if failed else ("unknown" if unknown else "yes")
    return status, unknown, failed
