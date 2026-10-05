"""Channel configuration for finance_mascot."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChannelConfig:
    channel_id: str = "finance_mascot"
    language: str = "English"
    channel_promise: str = "Understand money. Grow wealth."
    tone: str = "Friendly + educational"
    format: str = "YouTube Shorts"
    target_duration_seconds: int = 70
    mascot_asset_path: str = "assets/channels/finance_mascot/mascot.png"


CHANNEL = ChannelConfig()

