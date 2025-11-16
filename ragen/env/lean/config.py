from dataclasses import dataclass, field
from typing import Optional

@dataclass
class LeanEnvConfig:
    """Configuration options for the Lean theorem-proving environment."""

    dataset_name_or_path: str = field(
        default="CoderBak/minif2f",
        metadata={"description": "Hugging Face dataset identifier or local path."},
    )
    dataset_split: str = field(
        default="train",
        metadata={"description": "Split to use from the Hugging Face dataset."},
    )
    server_url: str = field(
        default="http://127.0.0.1:8000",
        metadata={"description": "Base URL for the Kimina Lean server."},
    )
    api_key: Optional[str] = field(
        default=None,
        metadata={"description": "Optional API key for authenticated servers."},
    )
    request_timeout: float = field(
        default=30.0,
        metadata={"description": "Timeout (seconds) when contacting the server."},
    )
    http_timeout: float = field(
        default=60.0,
        metadata={
            "description": "HTTP client timeout (seconds) for establishing connections."
        },
    )
    max_retries: int = field(
        default=3,
        metadata={"description": "Number of HTTP retry attempts when querying server."},
    )
    max_steps: int = field(
        default=100,
        metadata={"description": "Maximum number of tactics per episode."},
    )
    step_penalty: float = field(
        default=-0.1,
        metadata={"description": "Base reward applied each step."},
    )
    valid_step_reward: float = field(
        default=1.0,
        metadata={"description": "Additional reward for an accepted tactic."},
    )
    invalid_step_reward: float = field(
        default=-1.0,
        metadata={"description": "Additional reward when a tactic is rejected."},
    )
    success_reward: float = field(
        default=10.0,
        metadata={"description": "Reward granted when the proof is complete."},
    )
    timeout_penalty: float = field(
        default=-1.0,
        metadata={"description": "Penalty applied when the server times out."},
    )
    max_steps_penalty: float = field(
        default=-2.0,
        metadata={
            "description": "Penalty applied if max_steps is reached without success."
        },
    )
    default_imports: str = field(
        default="import Mathlib\nimport Aesop\nset_option maxHeartbeats 0\nopen BigOperators Real Nat Topology Rat",
        metadata={"description": "Imports prepended when dataset sample omits them."},
    )
    message_truncate_limit: int = field(
        default=0,
        metadata={"description": "Maximum characters for message (0 = no truncation)."},
    )
