"""Utility functions for WanVideo integration.

This module provides lazy loading for ComfyUI-WanVideoWrapper components
to avoid module loading order issues.
"""

from __future__ import annotations

import logging
import sys

# Get logger
logger = logging.getLogger("triple_ksampler.wvsampler.utils")

# Module-level cache for WanVideo unavailability
_WANVIDEO_UNAVAILABLE = False


def get_wanvideo_components():
    """Lazy loader for WanVideoSampler class and scheduler functions.

    Defers import until first use to avoid module loading order issues.
    ComfyUI loads custom_nodes in arbitrary order, so we can't rely on
    WanVideoWrapper being available at module import time.

    Returns:
        Tuple of (WanVideoSampler class, scheduler_list, get_scheduler function)

    Raises:
        ImportError: If WanVideoWrapper components cannot be found
    """
    global _WANVIDEO_UNAVAILABLE

    # If we already know WanVideo is unavailable, raise immediately
    if _WANVIDEO_UNAVAILABLE:
        raise ImportError("WanVideoSampler not available (cached)")

    # Use module-level cache to avoid repeated searches
    cache_key = "_wanvideo_cache"
    if hasattr(get_wanvideo_components, cache_key):
        return getattr(get_wanvideo_components, cache_key)

    # Search for the WanVideoSampler class in sys.modules
    # ComfyUI will have loaded the package, we just need to find it
    sampler_class = None
    for module_name, module in sys.modules.items():
        if "WanVideoWrapper" in module_name and "nodes_sampler" in module_name:
            try:
                sampler_class = module.WanVideoSampler
                break
            except AttributeError:
                continue

    # Fallback: Try direct import (for test environments where modules aren't pre-loaded)
    if sampler_class is None:
        try:
            import importlib
            import os

            # Find WanVideoWrapper in custom_nodes
            # Path: utils.py -> wvsampler/ -> triple_ksampler/ -> ComfyUI-TripleKSampler/ -> custom_nodes/
            current_dir = os.path.dirname(os.path.abspath(__file__))
            custom_nodes_dir = os.path.abspath(os.path.join(current_dir, "../../.."))
            wv_dir = os.path.join(custom_nodes_dir, "ComfyUI-WanVideoWrapper")
            wv_path = os.path.join(wv_dir, "nodes_sampler.py")

            if os.path.exists(wv_path):
                # Add custom_nodes directory to sys.path so package can be found
                if custom_nodes_dir not in sys.path:
                    sys.path.insert(0, custom_nodes_dir)

                # Import using importlib with hyphenated package name
                wv_package = importlib.import_module("ComfyUI-WanVideoWrapper.nodes_sampler")
                sampler_class = wv_package.WanVideoSampler
        except Exception:
            pass  # Silent fallback

    if sampler_class is None:
        # Cache unavailable state (message logged in __init__.py during registration)
        _WANVIDEO_UNAVAILABLE = True
        raise ImportError("WanVideoSampler not available")

    # Import scheduler components from WanVideoWrapper
    # Try multiple search patterns since Python may register modules differently
    search_patterns = [
        ("WanVideoWrapper", "schedulers", "__init__"),  # Original pattern
        ("WanVideoWrapper", "schedulers"),  # Without __init__
        ("wanvideo", "schedulers"),  # Lowercase variant
    ]

    scheduler_list = None
    get_scheduler_func = None
    for pattern in search_patterns:
        for module_name, module in sys.modules.items():
            if all(part in module_name for part in pattern):
                try:
                    scheduler_list = module.scheduler_list
                    get_scheduler_func = module.get_scheduler
                    break
                except AttributeError:
                    continue
        if scheduler_list is not None and get_scheduler_func is not None:
            break

    # Fallback: Try direct import of schedulers (for test environments)
    if scheduler_list is None or get_scheduler_func is None:
        try:
            import importlib
            import os

            # Find WanVideo schedulers in custom_nodes
            current_dir = os.path.dirname(os.path.abspath(__file__))
            custom_nodes_dir = os.path.abspath(os.path.join(current_dir, "../../.."))
            wv_dir = os.path.join(custom_nodes_dir, "ComfyUI-WanVideoWrapper")
            scheduler_path = os.path.join(wv_dir, "wanvideo", "schedulers", "__init__.py")

            if os.path.exists(scheduler_path):
                # Add custom_nodes directory to sys.path
                if custom_nodes_dir not in sys.path:
                    sys.path.insert(0, custom_nodes_dir)

                # Import schedulers module using importlib
                scheduler_module = importlib.import_module(
                    "ComfyUI-WanVideoWrapper.wanvideo.schedulers"
                )
                scheduler_list = scheduler_module.scheduler_list
                get_scheduler_func = scheduler_module.get_scheduler
        except Exception:
            pass  # Silent fallback

    if scheduler_list is None or get_scheduler_func is None:
        # Cache unavailable state (sampler already logged the message)
        _WANVIDEO_UNAVAILABLE = True
        raise ImportError("WanVideo scheduler components not available")

    result = (sampler_class, scheduler_list, get_scheduler_func)
    setattr(get_wanvideo_components, cache_key, result)
    return result
