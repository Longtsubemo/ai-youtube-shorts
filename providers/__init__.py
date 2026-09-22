"""Real (non-mock) provider implementations.

Providers live outside the `agents/` stage packages so stage logic never
imports an external SDK or makes a network call. A provider adapts a stage's
existing Protocol (docs/MODEL_STRATEGY.md); the stage stays SDK-free.
"""