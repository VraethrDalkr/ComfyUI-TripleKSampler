# ComfyUI-TripleKSampler

Advanced triple-stage sampling nodes for ComfyUI, specifically designed for Wan2.2 split models with Lightning LoRA integration.

## Overview

This package provides sophisticated triple-stage sampling workflow that implements:

1. **Base Denoising** - High-noise model processing with configurable CFG
2. **Lightning High Model** - Lightning LoRA high-model processing  
3. **Lightning Low Model** - Lightning LoRA low-model refinement

The nodes clone and patch models with sigma shift for optimal sampling without mutating the original models.

## Nodes

### TripleKSampler (Wan2.2-Lightning)

Main triple-stage sampler with optimized ease-of-use:

- **Streamlined Interface**: Essential parameters with smart defaults
- **Auto-computed base_steps**: Ensures base_steps * lightning_steps >= 20 for quality
- **Lightning-start Aware**: Auto-calculation accounts for when Lightning processing begins
- **Fixed lightning_start**: Always set to 1 for optimal workflow
- **Same Power**: Uses the same advanced algorithm as the Advanced node

### TripleKSampler Advanced (Wan2.2-Lightning)

Advanced variant with complete configurability:

- **Full Parameter Control**: Every aspect of the sampling process is configurable
- **5 Switching Strategies**: Manual step, manual boundary, T2V boundary, I2V boundary, 50% midpoint
- **Dynamic UI**: Parameters show/hide based on selected strategy for cleaner interface
- **Auto-calculation Options**: Both base_steps and switch_step can be auto-computed (-1) or manually set
- **Dry Run Mode**: Test configurations without actual sampling execution
- **Lightning-start Aware**: Auto-calculation accounts for when Lightning processing begins
- **Professional Features**: Advanced users get ultimate flexibility and testing capabilities

## Installation

1. Clone this repository to your ComfyUI custom_nodes directory:
   ```bash
   cd ComfyUI/custom_nodes/
   git clone https://github.com/VraethrDalkr/ComfyUI-TripleKSampler.git
   ```

2. Restart ComfyUI

3. The nodes will appear under `TripleKSampler/sampling` category

## Example Workflow

A complete example workflow is provided in [`example_workflow.json`](example_workflow.json) that demonstrates:

- **Side-by-side comparison** of both TripleKSampler nodes
- **Complete Wan2.2 T2V setup** with proper model loading
- **Lightning LoRA integration** for both HIGH and LOW models
- **Video generation workflow** with VAE decoding and video export

### Workflow Components

The example workflow includes:

1. **Model Setup:**
   - `wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors` (Base model)
   - `wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors` (Low noise model)
   - `Wan2.2-Lightning_T2V-v1.1-A14B-4steps-lora_HIGH_fp16.safetensors`
   - `Wan2.2-Lightning_T2V-v1.1-A14B-4steps-lora_LOW_fp16.safetensors`

2. **Sampling Comparison:**
   - **Main TripleKSampler:** Simplified interface with auto-computed parameters
   - **Advanced TripleKSampler:** Full parameter control for fine-tuning

3. **Video Output:**
   - 832x480 resolution, 41 frames
   - 16 FPS video export
   - Separate outputs for comparison

### Loading the Example

1. Download `example_workflow.json` from this repository
2. Open ComfyUI and drag the JSON file into the interface
3. Ensure you have the required Wan2.2 models and LoRAs
4. Queue the prompt to generate comparison videos

## Usage

### Basic Workflow

1. Load your Wan2.2 split models:
   - High-noise base model
   - Lightning high model (LightX2V)
   - Lightning low model (LightX2V)

2. Connect conditioning and latent inputs

3. Configure sampling parameters:
   - `shift`: Sigma shift (typically 5.0)
   - `base_cfg`: CFG for base denoising (typically 3.5)
   - `lightning_steps`: Total lightning steps (typically 8)

4. Choose model switching strategy:
   - **"50% of steps"**: Simple 50/50 split between lightning models
   - **"Manual switch step"**: Precise step-based control
   - **"T2V boundary"**: Auto-optimized for text-to-video (0.875)
   - **"I2V boundary"**: Auto-optimized for image-to-video (0.900)
   - **"Manual boundary"**: Custom sigma-based switching

5. Optional: Enable **Dry Run** mode (Advanced node) to test configurations without sampling

### Parameter Guidelines

- **shift**: 5.0 is recommended for most cases
- **base_cfg**: 3.5 works well for most prompts
- **lightning_steps**: 8 steps provide good quality/speed balance
- **boundary**: 0.875 for text-to-video, 0.900 for image-to-video

## Model Switching Strategies

The advanced node offers 5 switching strategies with dynamic UI that shows only relevant parameters:

### 1. "50% of steps" (Auto-Midpoint)
Simple approach that switches at 50% of lightning steps. Reliable and straightforward.

### 2. "Manual switch step"
Allows precise control over the switching step. Shows switch_step parameter for manual configuration.

### 3. "T2V boundary" (Auto-Boundary)
Optimized for text-to-video models, automatically uses boundary value of 0.875 for sigma-based switching.

### 4. "I2V boundary" (Auto-Boundary)
Optimized for image-to-video models, automatically uses boundary value of 0.900 for sigma-based switching.

### 5. "Manual boundary"
Full control over sigma boundary value. Shows switch_boundary parameter for manual configuration.

**Dynamic UI**: Only relevant parameters are shown based on the selected strategy, keeping the interface clean and focused.

## Configuration

The package includes a `constants.py` file with user-configurable parameters:

```python
# Quality threshold for automatic base step calculation
MIN_TOTAL_STEPS = 20

# Default sigma boundaries for different model types
DEFAULT_BOUNDARY_T2V = 0.875  # Text-to-video models
DEFAULT_BOUNDARY_I2V = 0.900  # Image-to-video models

# Logging configuration
LOG_LEVEL = "INFO"  # DEBUG shows detailed internal calculations, INFO shows essential workflow info
```

Users can modify these values to tune sampling behavior without changing the core implementation.

## Auto-Calculation Methods

When you set `base_steps=-1`, the nodes automatically calculate optimal values using one of three methods:

### Simple Math
**When**: `lightning_start=1` (most common case)
**Behavior**: Direct mathematical calculation that guarantees perfect alignment between Stage 1 and Stage 2 transition points
**Result**: Always achieves perfect stage alignment with minimal computation

### Mathematical Search
**When**: `lightning_start>1` (advanced configurations)
**Behavior**: Searches through possible values within constraints to find perfect alignment
**Result**: Usually achieves perfect stage alignment, highly efficient search algorithm

### Fallback
**When**: No perfect alignment can be found within search constraints (very rare)
**Behavior**: Uses approximation to get as close as possible to optimal alignment
**Result**: Near-optimal alignment when perfect alignment is mathematically impossible

**Log Messages**: You'll see these method names in the console output when using auto-calculation, helping you understand which approach was used for your specific configuration.

## Advanced Features

### Dry Run Mode (Advanced Node Only)
The advanced node includes a **Dry Run** boolean parameter for testing configurations:

- **Purpose**: Test parameter combinations without expensive sampling operations
- **Validation**: Performs complete parameter validation and logging
- **Performance**: Instant feedback for workflow testing and debugging
- **Usage**: Enable the "dry_run" checkbox in the advanced node interface

### Clean Architecture
Built with professional software engineering principles:

- **Inheritance Hierarchy**: Base → Advanced → Simple node structure eliminates code duplication
- **Dynamic UI**: Interface adapts to selected strategy for optimal user experience
- **Comprehensive Validation**: Edge cases and parameter conflicts are handled gracefully

## Logging

The nodes provide detailed logging including:
- Stage execution with step ranges and denoising percentages
- Model switching strategy and computed switch points
- Auto-computed parameter values (both nodes when applicable)
- Stage overlap warnings with actionable suggestions

**Log Levels**: Set `LOG_LEVEL="DEBUG"` in `constants.py` for detailed internal calculations, or `LOG_LEVEL="INFO"` (default) for essential workflow information. Warning and error messages always appear regardless of the log level setting.

## Development

### Requirements
- ComfyUI
- PyTorch
- Standard ComfyUI dependencies

### Code Structure
- `__init__.py`: Package initialization, node registration, and web directory setup
- `triple_ksampler_wan22.py`: Main implementation with clean inheritance hierarchy:
  - `TripleKSamplerWan22Base`: Shared functionality and core sampling logic
  - `TripleKSamplerWan22LightningAdvanced`: Advanced node with full parameter control
  - `TripleKSamplerWan22Lightning`: Simple node with streamlined interface
- `constants.py`: Configuration constants and user-tunable parameters
- `web/triple_ksampler_ui.js`: JavaScript for dynamic UI parameter visibility

### Testing
To test the nodes are properly loaded:
1. Restart ComfyUI
2. Check console for any import errors
3. Verify nodes appear in the node browser under `TripleKSampler/sampling`

## Version History

### v0.3.2 (Current)
- **FEATURE**: Dry Run Mode - Boolean parameter in advanced node for testing configurations without sampling execution
- **REMOVAL**: KJNodes compatibility system removed (no longer needed - KJNodes fixed their transformer_options issue)
- **ENHANCEMENT**: Improved type annotations with Optional[str] for better IDE support
- **SIMPLIFICATION**: Removed ENABLE_KJNODES_COMPATIBILITY_FIX constant and ~79 lines of compatibility code
- **ARCHITECTURE**: Continued refinement of clean inheritance structure

### v0.3.1
- **ARCHITECTURE**: Major inheritance refactor - Base → Advanced → Simple eliminates ~400 lines of code duplication
- **BREAKING**: Standardized error handling with consistent 'fail fast' approach
- **Enhanced Error Messages**: All configuration conflicts now raise ValueError with actionable guidance
- **Code Quality**: Simplified line break usage, enhanced parameter validation with edge case coverage

### v0.3.0
- **BREAKING**: Parameter names simplified for clarity (`switching_strategy` → `switch_strategy`, `midpoint` → `switch_step`, `boundary` → `switch_boundary`)
- **BREAKING**: Strategy option renamed ("50% of lightning steps" → "50% of steps")
- **Enhanced Validation**: Comprehensive edge case validation and error handling
- **Logging Improvements**: Clean visual separators with bare logger for empty lines
- **Code Refactoring**: Simplified internal variable naming and stage execution logic
- **Bug Fixes**: Proper noise addition for Stage3-only scenarios
- **UI Improvements**: Dynamic UI restricted to advanced node only, lightning_cfg removed from simple node
- **File Organization**: Renamed workflow_example.json to example_workflow.json

### v0.2.0
- **BREAKING**: Node names swapped for better UX - Simple variant is now main "TripleKSampler"
- **Enhanced Advanced Node**: Auto-calculation options for both base_steps and switch_step (-1 for auto)
- **Lightning-start Awareness**: Auto-calculation accounts for lightning processing timing
- **Configuration File**: Moved constants to separate `constants.py` with proper documentation
- **Quality Threshold**: MIN_TOTAL_STEPS set to 20 for optimal quality/performance balance

### v0.1.0
- Initial release with configurable minimum total steps constant
- Comprehensive triple-stage sampling implementation
- Both full and simplified node variants
- Professional code structure and documentation

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Support

If you encounter issues or have questions:
- Check the [Issues](https://github.com/VraethrDalkr/ComfyUI-TripleKSampler/issues) page
- Create a new issue with detailed description and ComfyUI console output

## Author

**VraethrDalkr** - [GitHub Profile](https://github.com/VraethrDalkr)