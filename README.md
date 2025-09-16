# ComfyUI-TripleKSampler

Triple-stage sampling nodes for ComfyUI, specifically designed for Wan2.2 split models with Lightning LoRA integration.

## Overview

This package provides triple-stage sampling workflow that implements:

1. **Base Denoising** - High-noise model processing with configurable CFG
2. **Lightning High Model** - Lightning LoRA high-model processing  
3. **Lightning Low Model** - Lightning LoRA low-model refinement

The nodes clone and patch models with sigma shift for optimal sampling without mutating the original models.

## Nodes

### TripleKSampler (Wan2.2-Lightning)

Main triple-stage sampler with streamlined interface:

- **Streamlined Interface**: Essential parameters with smart defaults
- **Auto-computed base_steps**: Ensures base_steps * lightning_steps >= 20 for quality
- **Lightning-start Aware**: Auto-calculation accounts for when Lightning processing begins
- **Fixed lightning_start**: Always set to 1 for optimal workflow
- **Same Functionality**: Uses the same algorithm as the Advanced node

### TripleKSampler Advanced (Wan2.2-Lightning)

Variant with complete configurability:

- **Full Parameter Control**: Every aspect of the sampling process is configurable
- **5 Switching Strategies**: Manual step, manual boundary, T2V boundary, I2V boundary, 50% midpoint
- **Dynamic UI**: Parameters show/hide based on selected strategy for cleaner interface
- **Auto-calculation Options**: Both base_steps and switch_step can be auto-computed (-1) or manually set
- **Dry Run Mode**: Test configurations without actual sampling execution
- **Lightning-start Aware**: Auto-calculation accounts for when Lightning processing begins
- **Full Control**: Complete flexibility and testing capabilities

## Installation

1. Clone this repository to your ComfyUI custom_nodes directory:
   ```bash
   cd ComfyUI/custom_nodes/
   git clone https://github.com/VraethrDalkr/ComfyUI-TripleKSampler.git
   ```

2. Install requirements (for TOML configuration support):
   ```bash
   cd ComfyUI-TripleKSampler
   pip install -r requirements.txt
   ```

3. Restart ComfyUI

4. The nodes will appear under `TripleKSampler/sampling` category

## Example Workflows

Example workflows are included: `example_workflow_t2v.json` and `example_workflow_i2v.json`

## Usage

### Basic Workflow

1. Load your Wan2.2 split models:
   - Base high-noise model
   - Lightning high-noise model
   - Lightning low-noise model

2. Connect conditioning and latent inputs

3. Configure sampling parameters:
   - `shift`: Sigma shift (typically 5.0)
   - `base_cfg`: CFG for base denoising (typically 3.5)
   - `lightning_steps`: Total lightning steps (typically 8)

4. Choose model switching strategy:
   - **"50% of steps"**: Simple 50/50 split between lightning models
   - **"Manual switch step"**: Precise step-based control
   - **"T2V boundary"**: Auto-configured for text-to-video (0.875)
   - **"I2V boundary"**: Auto-configured for image-to-video (0.900)
   - **"Manual boundary"**: Custom sigma-based switching

5. Optional: Enable **Dry Run** mode (Advanced node) to test configurations without sampling

### Parameter Guidelines

- **shift**: 5.0 is recommended for most cases
- **base_cfg**: 3.5 works well for most prompts
- **lightning_steps**: 8 steps provide good quality/speed balance
- **boundary**: 0.875 for text-to-video, 0.900 for image-to-video

## Model Switching Strategies

The Advanced node offers 5 switching strategies with dynamic UI that shows only relevant parameters:

### 1. "50% of steps" (Auto-Midpoint)
Simple approach that switches at 50% of lightning steps. Reliable and straightforward.

### 2. "Manual switch step"
Allows precise control over the switching step. Shows switch_step parameter for manual configuration.

### 3. "T2V boundary" (Auto-Boundary)
Configured for text-to-video models, automatically uses boundary value of 0.875 for sigma-based switching.

### 4. "I2V boundary" (Auto-Boundary)
Configured for image-to-video models, automatically uses boundary value of 0.900 for sigma-based switching.

### 5. "Manual boundary"
Full control over sigma boundary value. Shows switch_boundary parameter for manual configuration.

**Dynamic UI**: Only relevant parameters are shown based on the selected strategy, keeping the interface clean and focused.

## Configuration

The package uses a TOML-based configuration system that avoids git conflicts when updating. Configuration priority:

1. **`config.toml`** (user's custom settings, gitignored)
2. **`config.example.toml`** (template with defaults, tracked in git)
3. **Hardcoded defaults** (final fallback)

### Setup

**Automatic Setup (Recommended):**
1. Install the node and restart ComfyUI
2. The node automatically creates `config.toml` from the template on first run
3. Edit `config.toml` with your preferred values
4. Restart ComfyUI to apply changes

**Manual Setup (Optional):**
1. Copy the template: `cp config.example.toml config.toml`
2. Edit `config.toml` with your preferred values
3. Restart ComfyUI to apply changes

### Configuration Options

```toml
[sampling]
min_total_steps = 20

[boundaries]
default_t2v = 0.875
default_i2v = 0.900

[logging]
level = "INFO"
```

**Parameters:**
- **`min_total_steps`**: Quality threshold for automatic base step calculation
- **`default_t2v`**: Default sigma boundary for text-to-video models (0.875)
- **`default_i2v`**: Default sigma boundary for image-to-video models (0.900)
- **`level`**: Logging level (`"DEBUG"` shows detailed calculations, `"INFO"` shows essential workflow info)

The `config.toml` file is automatically created on first run and is gitignored to prevent conflicts during updates.

## Auto-Calculation Methods

When you set `base_steps=-1`, the nodes automatically calculate optimal values using one of three methods:

### Simple Math
**When**: `lightning_start=1` (most common case)
**Behavior**: Direct mathematical calculation that guarantees perfect alignment between Stage 1 and Stage 2 transition points
**Result**: Achieves stage alignment with minimal computation

### Mathematical Search
**When**: `lightning_start>1` (complex configurations)
**Behavior**: Searches through possible values within constraints to find perfect alignment
**Result**: Usually achieves stage alignment, efficient search algorithm

### Fallback
**When**: No perfect alignment can be found within search constraints (very rare)
**Behavior**: Uses approximation to get as close as possible to optimal alignment
**Result**: Good alignment when exact alignment is mathematically impossible

**Log Messages**: You'll see these method names in the console output when using auto-calculation, helping you understand which approach was used for your specific configuration.

## Additional Features

### Dry Run Mode (Advanced Node Only)
The Advanced node includes a **Dry Run** boolean parameter for testing configurations:

- **Purpose**: Test parameter combinations without expensive sampling operations
- **Validation**: Performs complete parameter validation and logging
- **Performance**: Instant feedback for workflow testing and debugging
- **Usage**: Enable the "dry_run" checkbox in the Advanced node interface


## Logging

The nodes provide detailed logging including:
- Stage execution with step ranges and denoising percentages
- Model switching strategy and computed switch points
- Auto-computed parameter values (both nodes when applicable)
- Stage overlap warnings with actionable suggestions

**Log Levels**: Set `level = "DEBUG"` in `config.toml` for detailed internal calculations, or `level = "INFO"` (default) for essential workflow information. Warning and error messages always appear regardless of the log level setting.

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
- `config.example.toml`: Configuration template with default values
- `web/triple_ksampler_ui.js`: JavaScript for dynamic UI parameter visibility

### Testing
To test the nodes are properly loaded:
1. Restart ComfyUI
2. Check console for any import errors
3. Verify nodes appear in the node browser under `TripleKSampler/sampling`


## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.


## Support

If you encounter issues or have questions:
- Check the [Issues](https://github.com/VraethrDalkr/ComfyUI-TripleKSampler/issues) page
- Create a new issue with detailed description and ComfyUI console output

## Author

**VraethrDalkr** - [GitHub Profile](https://github.com/VraethrDalkr)