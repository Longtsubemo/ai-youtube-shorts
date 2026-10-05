"""Simplified script validator."""

from __future__ import annotations


class ScriptValidation:
    def __init__(self, issues=None, passed=True):
        self.issues = issues or []
        self.passed = passed

    def to_dict(self):
        return {"issues": self.issues, "passed": self.passed}



def validate_script(*args, **kwargs):
    return ScriptValidation()



def validate_script_from_package(*args, **kwargs):
    return ScriptValidation()

