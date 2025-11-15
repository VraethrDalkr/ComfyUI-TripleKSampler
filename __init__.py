"""
ComfyUI-TripleKSampler: Advanced triple-stage sampling for Wan2.2 split models.

This package provides custom nodes for ComfyUI that implement a sophisticated
triple-stage sampling workflow optimized for Wan2.2 split models with Lightning LoRA.
"""

import os
import sys
from pathlib import Path

# Add package directory to Python path so triple_ksampler module can be found
# This is necessary because ComfyUI loads custom nodes without installing them
package_dir = os.path.dirname(os.path.abspath(__file__))
if package_dir not in sys.path:
    sys.path.insert(0, package_dir)

from .ksampler_nodes import (
    NODE_CLASS_MAPPINGS as KSAMPLER_MAPPINGS,
)
from .ksampler_nodes import (
    NODE_DISPLAY_NAME_MAPPINGS as KSAMPLER_DISPLAY_MAPPINGS,
)

# Case-insensitive search for ComfyUI-WanVideoWrapper directory
custom_nodes_dir = Path(__file__).parent.parent
wanvideo_dir = None
target_name = "comfyui-wanvideowrapper"

if custom_nodes_dir.exists():
    for item in custom_nodes_dir.iterdir():
        if item.is_dir() and item.name.lower() == target_name:
            wanvideo_dir = item
            break

# Verify WanVideoWrapper is properly installed by checking for required files
# This prevents crashes if directory exists but WanVideoSampler is unavailable
if wanvideo_dir is not None:
    required_files = [
        wanvideo_dir / "__init__.py",
        wanvideo_dir / "nodes_sampler.py",  # Contains WanVideoSampler class
        wanvideo_dir / "wanvideo" / "schedulers" / "__init__.py",  # Contains scheduler functions
    ]
    wanvideo_available = all(f.exists() for f in required_files)
else:
    wanvideo_available = False

# Import WanVideo wrapper nodes (optional - will fail gracefully if ComfyUI-WanVideoWrapper not installed)
if not wanvideo_available:
    # WanVideoWrapper not installed - skip import entirely
    NODE_CLASS_MAPPINGS = KSAMPLER_MAPPINGS
    NODE_DISPLAY_NAME_MAPPINGS = KSAMPLER_DISPLAY_MAPPINGS
    print(
        "[TripleKSampler] INFO: ComfyUI-WanVideoWrapper not installed - TripleWVSampler nodes unavailable"
    )
    print(
        "[TripleKSampler] INFO: Only KSampler nodes will be registered. Install ComfyUI-WanVideoWrapper for WanVideo support."
    )
else:
    # Directory exists - try import (NO verification call to avoid load order issues)
    try:
        from .wvsampler_nodes import (
            NODE_CLASS_MAPPINGS as WANVIDEO_MAPPINGS,
        )
        from .wvsampler_nodes import (
            NODE_DISPLAY_NAME_MAPPINGS as WANVIDEO_DISPLAY_MAPPINGS,
        )

        # Merge mappings
        NODE_CLASS_MAPPINGS = {**KSAMPLER_MAPPINGS, **WANVIDEO_MAPPINGS}
        NODE_DISPLAY_NAME_MAPPINGS = {**KSAMPLER_DISPLAY_MAPPINGS, **WANVIDEO_DISPLAY_MAPPINGS}

        # Success message (matching commit 2e51d5e style)
        print("[TripleKSampler] ✓ WanVideo wrapper available - TripleWVSampler nodes registered")

    except ImportError as e:
        # Import failed even though directory exists
        NODE_CLASS_MAPPINGS = KSAMPLER_MAPPINGS
        NODE_DISPLAY_NAME_MAPPINGS = KSAMPLER_DISPLAY_MAPPINGS
        print(f"[TripleKSampler] ⚠️ ComfyUI-WanVideoWrapper found but failed to load: {e}")
        print(
            "[TripleKSampler] INFO: Only KSampler nodes will be registered. Check WanVideoWrapper installation."
        )

__version__ = "0.10.5"
__author__ = "VraethrDalkr"
__description__ = "Triple-stage KSampler for Wan2.2 split models with Lightning LoRA"

# ComfyUI node registration
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
WEB_DIRECTORY = "./web"
