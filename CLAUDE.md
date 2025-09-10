# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ComfyUI-TripleKSampler is a custom node package for ComfyUI that implements advanced triple-stage sampling for Wan2.2 split models with Lightning LoRA. The package provides two main node classes with sophisticated sampling workflows.

## Development Commands

### Testing Node Structure
```bash
# Test Python syntax compilation
python3 -m py_compile triple_ksampler_wan22.py

# Test in ComfyUI environment (requires ComfyUI to be running)
# The nodes will appear under "TripleKSampler/sampling" category in ComfyUI
```

### Git Operations
```bash
# Initialize repository (already done)
git init
git add .
git commit -m "Initial commit"

# Standard git workflow
git add .
git commit -m "Description of changes"
```

## Code Architecture

### Package Structure
- `__init__.py`: Package initialization, exports node mappings for ComfyUI registration
- `triple_ksampler_wan22.py`: Main implementation containing both node classes
- `README.md`: User documentation
- `.gitignore`: Git ignore patterns for Python, PyTorch, and ComfyUI

### Key Components

#### TripleKSamplerWan22Lightning (Full Node)
- Complete parameter control for all sampling stages
- Supports both midpoint and sigma boundary model switching
- Comprehensive logging with denoising percentage calculations
- Proper error handling and parameter validation

#### SimpleTripleKSamplerWan22Lightning (Simplified Node)  
- Streamlined interface with auto-computed parameters
- Fixed lightning_start=1, auto-computed base_steps
- Delegates to full implementation for consistency

### Core Methods
- `_run_sampling_stage()`: Executes individual sampling stages using KSamplerAdvanced
- `_compute_boundary_switching_step()`: Calculates sigma-based model switching
- `_format_stage_range()`: Generates human-readable logging messages
- `sample()`: Main entry point for triple-stage sampling workflow

### ComfyUI Integration
- Node registration via `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS`
- Proper input type definitions with tooltips and validation
- Category organization under "TripleKSampler/sampling"

## Dependencies

### Required ComfyUI Modules
- `comfy.model_sampling`: For sigma calculations and model sampling
- `comfy.samplers`: For sampler and scheduler access
- `nodes`: For KSamplerAdvanced functionality
- `comfy_extras.nodes_model_advanced.ModelSamplingSD3`: For model patching

### Python Requirements
- PyTorch (torch): For tensor operations
- Standard library: logging, math, typing

## Code Style and Standards

### Followed Conventions
- PEP8 compliance with proper line length and formatting
- Comprehensive docstrings using Google style
- Type hints throughout for better code documentation
- Descriptive method and variable names
- Proper error handling with informative messages

### Logging Strategy
- Module-level logger with "[TripleKSampler]" prefix
- INFO level logging for user-relevant sampling stage information
- Detailed stage execution with step ranges and percentages
- Model switching strategy logging

## Development Notes

### Testing Approach
- Syntax validation via py_compile
- Integration testing requires full ComfyUI environment
- Nodes should appear in ComfyUI node browser after restart

### Model Patching Strategy
- Always clone models before patching (no mutation)
- Use ModelSamplingSD3 for sigma shift application
- Separate patched instances for each model type

### Parameter Validation
- Range checking for all numeric inputs
- Logical validation (e.g., lightning_start < lightning_steps)
- Clear error messages for invalid configurations

### Future Considerations
- The `_ENABLE_CONSISTENCY_CHECK` toggle for development debugging
- Extensible architecture for additional sampling strategies
- Modular design allows easy addition of new node variants