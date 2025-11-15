"""Toast notification utilities for TripleKSampler.

This module encapsulates ComfyUI's PromptServer integration for sending
toast notifications to the UI. All operations gracefully handle the case
where PromptServer is not available (e.g., during testing or if server
import fails).

Toast notifications are used to communicate important information to users:
- Dry-run validation results
- Overlap warnings between sampling stages
- Configuration recommendations
"""

import os
from typing import Any

# Toast notification durations (milliseconds)
TOAST_LIFE_OVERLAP = 8000  # Overlap warnings
TOAST_LIFE_DRY_RUN = 12000  # Dry run results

# Import PromptServer with graceful fallback
try:
    if os.environ.get("COMFYUI_TESTING", "0") == "1":
        # Skip server import in testing mode
        PromptServer = None
    else:
        from server import PromptServer
except ImportError:
    # Fallback if server import fails
    PromptServer = None


def send_toast_notification(
    message_type: str,
    severity: str,
    summary: str,
    detail: str,
    life: int | None = None,
) -> bool:
    """Send toast notification to ComfyUI UI via PromptServer.

    Safely sends a toast message to the ComfyUI frontend if PromptServer is
    available. Gracefully handles the case where PromptServer is not available
    (returns False).

    Args:
        message_type: Message type identifier for the frontend handler
            (e.g., "triple_ksampler_dry_run", "triple_ksampler_overlap")
        severity: Severity level - "info", "warn", or "error"
        summary: Short summary text shown prominently in the toast
        detail: Detailed message text shown in the toast body
        life: Toast duration in milliseconds (None uses frontend default)

    Returns:
        bool: True if notification was sent successfully, False otherwise

    Example:
        >>> send_toast_notification(
        ...     "triple_ksampler_dry_run",
        ...     "info",
        ...     "Dry Run Complete",
        ...     "Stage 1: 10 steps\\nStage 2: 5 steps\\nStage 3: 3 steps",
        ...     life=TOAST_LIFE_DRY_RUN
        ... )
        True
    """
    try:
        if PromptServer and hasattr(PromptServer, "instance") and PromptServer.instance:
            message_data: dict[str, Any] = {
                "severity": severity,
                "summary": summary,
                "detail": detail,
            }
            if life is not None:
                message_data["life"] = life

            PromptServer.instance.send_sync(message_type, message_data)
            return True
    except Exception:
        # Silently fail if PromptServer is not available or send fails
        pass

    return False


def send_dry_run_notification(detail: str) -> bool:
    """Send dry-run completion toast notification.

    Convenience function for sending dry-run validation results to the UI.

    Args:
        detail: Detailed validation results (multi-line string with stage info)

    Returns:
        bool: True if notification was sent successfully, False otherwise

    Example:
        >>> detail = "Stage 1: base → steps 0-10\\nStage 2: lightning_high → steps 10-15"
        >>> send_dry_run_notification(detail)
        True
    """
    return send_toast_notification(
        message_type="triple_ksampler_dry_run",
        severity="info",
        summary="TripleKSampler: Dry Run Complete",
        detail=detail,
        life=TOAST_LIFE_DRY_RUN,
    )


def send_overlap_warning(overlap_pct: float) -> bool:
    """Send stage overlap warning toast notification.

    Convenience function for warning users about overlapping stages (Stage 1
    end > Stage 2 start), which can indicate configuration issues.

    Args:
        overlap_pct: Percentage of overlap between stages

    Returns:
        bool: True if notification was sent successfully, False otherwise

    Example:
        >>> send_overlap_warning(25.5)
        True
    """
    detail = (
        f"Stage 1 and Stage 2 overlap by {overlap_pct:.1f}%. "
        "Consider base_steps=-1 or adjust lightning parameters."
    )
    return send_toast_notification(
        message_type="triple_ksampler_overlap",
        severity="warn",
        summary="TripleKSampler: Stage overlap",
        detail=detail,
        life=TOAST_LIFE_OVERLAP,
    )
