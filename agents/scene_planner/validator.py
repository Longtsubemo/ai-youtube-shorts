"""Simplified validator."""

from __future__ import annotations


class ScenePlanningValidation:
    def __init__(self, issues=None, passed=True):
        self.issues = issues or []
        self.passed = passed

    def to_dict(self):
        return {"issues": self.issues, "passed": self.passed}


class ValidationIssue:
    def __init__(self, code=None, message=None):
        self.code = code
        self.message = message

    def to_dict(self):
        return {'code': self.code, 'message': self.message}

def validate_production_plan(*args, **kwargs):
    return ScenePlanningValidation()

