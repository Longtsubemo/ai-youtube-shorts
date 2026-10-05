"""Channel configuration for finance_mascot.

Single source of truth for channel *identity*. Stage 1 (topic selection)
consumes this so candidates are generated for the right channel rather than
for a hard-coded generic set of categories.

The mascot asset is configured by path only. The asset itself does not exist
yet and is never generated from this repository.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MascotConfig:
    """The recurring on-screen character.

    The mascot is a financial detective. It appears alongside the narration
    and does not speak.
    """

    name: str = "fox"
    role: str = "financial_detective"
    asset_path: str = "assets/channels/finance_mascot/mascot.png"
    asset_exists: bool = False
    speaks: bool = False
    note: str = (
        "Appears alongside the narration as a visual presence. "
        "Never speaks and never narrates."
    )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "role": self.role,
            "asset_path": self.asset_path,
            "asset_exists": self.asset_exists,
            "speaks": self.speaks,
            "note": self.note,
        }


@dataclass(frozen=True)
class ChannelConfig:
    """Who this channel is for and what it promises."""

    channel_id: str = "finance_mascot"
    language: str = "English"
    channel_promise: str = "Understand money. Grow wealth."
    audience: str = (
        "Broad English-speaking audience interested in money and wealth."
    )
    tone: str = "Friendly + educational"
    format: str = "YouTube Shorts"
    target_duration_seconds: int = 70
    mascot: MascotConfig = field(default_factory=MascotConfig)

    @property
    def mascot_asset_path(self) -> str:
        """Configurable location of the mascot asset.

        Kept as a property for backwards compatibility with callers that read
        ``CHANNEL.mascot_asset_path``. The asset is NOT created here.
        """
        return self.mascot.asset_path

    def to_dict(self) -> dict:
        return {
            "channel_id": self.channel_id,
            "language": self.language,
            "channel_promise": self.channel_promise,
            "audience": self.audience,
            "tone": self.tone,
            "format": self.format,
            "target_duration_seconds": self.target_duration_seconds,
            "mascot": self.mascot.to_dict(),
        }


CHANNEL = ChannelConfig()