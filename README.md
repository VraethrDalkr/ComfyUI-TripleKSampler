# ComfyUI-TripleKSampler

Triple-stage sampling nodes for ComfyUI, specifically designed for Wan2.2 split models with Lightning LoRA integration.

## Overview

This package provides triple-stage sampling workflow that implements:

1. **Base Denoising** - High-noise model processing with configurable CFG
2. **Lightning High Model** - Lightning LoRA high-model processing  
3. **Lightning Low Model** - Lightning LoRA low-model refinement

The nodes clone and patch models with sigma shift for optimal sampling without mutating the original models.

## Why TripleKSampler vs Multiple KSamplers?

When working with Wan2.2 and Lightning LoRA, users often create workflows with multiple KSampler nodes. TripleKSampler offers a different approach to step allocation across the three stages.

### Step Resolution Approaches (Example Comparison)

**Example Multi-KSampler Setup:**
```
Total steps: 8 for all stages
├── Base High: steps 0-2 of 8 (25% denoising)
├── Lightning High: steps 2-4 of 8 (25% denoising)
└── Lightning Low: steps 4-8 of 8 (50% denoising)
```

**Example TripleKSampler Approach:**
```
Different step resolution per stage purpose
├── Base High: steps 0-5 of 20 (25% denoising)
├── Lightning High: steps 2-4 of 8 (25% denoising)
└── Lightning Low: steps 4-8 of 8 (50% denoising)
```

*These examples illustrate the design philosophy difference between approaches.*

### Design Philosophy

TripleKSampler separates step resolution from denoising percentage. The base model receives higher step resolution while Lightning stages use their native low steps schedule. This approach considers that base models were designed for longer step sequences, while Lightning LoRAs are optimized for fewer steps.

### What TripleKSampler Handles

- **Step Calculation**: Automatically determines optimal base model steps (see Auto-Calculation Methods section)
- **Stage Transitions**: Ensures perfect denoising alignment without gaps or overlaps
- **Multi-Model Coordination**: Manages three different models with their respective step requirements
- **Configuration Validation**: Prevents parameter conflicts and stage errors

The node automates the complex coordination needed for three-model workflows while maintaining proper denoising coverage across all stages.

## Key Differences from Native KSampler

TripleKSampler implements a triple-stage architecture for Wan2.2 split models with Lightning LoRA:

### Architecture
- **Native KSampler**: Single-stage sampling with one model
- **TripleKSampler**: Three-stage progression: Base → Lightning High → Lightning Low

### Model Processing
- **Native KSampler**: Operates directly on provided models
- **TripleKSampler**: Uses ModelSamplingSD3 internally for sigma shift application

### Parameter Management
- **Native KSampler**: Manual step count with fixed CFG
- **TripleKSampler**: Auto-calculation of stage transitions with separate CFG control per stage

### Parameters
- **TripleKSampler**: Specialized parameters for multi-stage workflow and Lightning LoRA integration

### Use Case
- **Native KSampler**: General-purpose single-model sampling
- **TripleKSampler**: Wan2.2 + Lightning LoRA workflows

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

## Nodes

### TripleKSampler (Wan2.2-Lightning)

Main triple-stage sampler with streamlined interface:

- **Streamlined Interface**: Essential parameters with smart defaults
- **Auto-computed base_steps**: Uses quality threshold-based calculation for optimal base model utilization
- **Lightning-start Aware**: Auto-calculation accounts for when Lightning processing begins
- **Configurable lightning_start**: Defaults to 1 but can be adjusted for different workflows
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

## Example Workflows

Example workflows are included in the `example_workflows/` directory: `t2v_workflow.json` and `i2v_workflow.json`

## Usage

### Basic Workflow

1. Load your Wan2.2 split models:
   - Base high-noise model
   - Lightning high-noise model
   - Lightning low-noise model

2. Connect conditioning and latent inputs

3. Configure sampling parameters:
   - `sigma_shift`: Sigma shift value (default: 5.0)
   - `base_cfg`: CFG for base denoising (default: 3.5)
   - `lightning_steps`: Total lightning steps (default: 8)

4. Choose model switching strategy (see Model Switching Strategies section for details)

5. Optional: Enable **Dry Run** mode (Advanced node) to test configurations without sampling

## Parameter Reference

Parameters unique to TripleKSampler compared to native KSampler:

### Core TripleKSampler Parameters

**sigma_shift**
- Applies ModelSamplingSD3 for sigma shift

**base_steps** (Advanced node only)
- Number of steps for Stage 1 (base high-noise model)
- Use -1 for auto-calculation based on quality threshold
- Auto-calculation ensures optimal transition between stages

**lightning_start**
- Starting step within the lightning schedule
- Set to 0 to skip Stage 1 entirely
- Default is 1 for standard three-stage workflow

**lightning_steps**
- Total steps for the lightning sampling process
- Default is 8 in our nodes
- Controls the resolution of the denoising schedule

**base_cfg**
- CFG scale for Stage 1 only
- Default is 3.5 in our nodes
- Separate from lightning_cfg for per-stage control

**lightning_cfg** (Advanced node only)
- CFG scale for Stage 2 and Stage 3
- In the regular node, automatically set to 1.0
- Independent control from base_cfg

### Switching Strategies

**switching_strategy**
- Controls transition between lightning high and low models
- Options: "50% of steps", "Manual switch step", "T2V boundary", "I2V boundary", "Manual boundary"
- Manual modes ("Manual switch step", "Manual boundary") available in Advanced node only

**switch_step** (Advanced node only)
- Manual step number for switching (when using "Manual switch step")
- Use -1 for auto-calculation at 50% of lightning steps

**switch_boundary** (Advanced node only)
- Sigma boundary value for switching (when using "Manual boundary")
- Defaults to 0.875
- T2V boundary strategy uses 0.875, I2V boundary strategy uses 0.900 (can be changed in config.toml)

### Parameter Guidelines

- **sigma_shift**: Adjust based on your specific models and use case
- **base_cfg**: Experiment with different values based on your prompt
- **lightning_steps**: Balance between quality and speed for your needs
- **boundary**: 0.875 for text-to-video, 0.900 for image-to-video

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
base_quality_threshold = 20

[boundaries]
default_t2v = 0.875
default_i2v = 0.900

[logging]
level = "INFO"
```

**Parameters:**
- **`base_quality_threshold`**: Minimum total steps constraint for base_steps auto-calculation
- **`default_t2v`**: Default sigma boundary for text-to-video models (0.875)
- **`default_i2v`**: Default sigma boundary for image-to-video models (0.900)
- **`level`**: Logging level (`"DEBUG"` shows detailed calculations, `"INFO"` shows essential workflow info)

The `config.toml` file is automatically created on first run and is gitignored to prevent conflicts during updates.

## Model Switching Strategies

The Advanced node offers 5 switching strategies with dynamic UI that shows only relevant parameters:

### 1. "50% of steps" (Auto-Midpoint)
Simple approach that switches at 50% of lightning steps (rounded up). Reliable and straightforward.

### 2. "Manual switch step"
Allows precise control over the switching step. Shows switch_step parameter for manual configuration.

### 3. "T2V boundary" (Auto-Boundary)
Configured for text-to-video models, automatically uses boundary value of 0.875 for sigma-based switching.

### 4. "I2V boundary" (Auto-Boundary)
Configured for image-to-video models, automatically uses boundary value of 0.900 for sigma-based switching.

### 5. "Manual boundary"
Full control over sigma boundary value. Shows switch_boundary parameter for manual configuration.

**Dynamic UI**: Only relevant parameters are shown based on the selected strategy, keeping the interface clean and focused.

## Auto-Calculation Methods

The nodes provide intelligent auto-calculation for optimal parameter values:

### Base Steps Auto-Calculation
When you set `base_steps=-1` (Advanced node only), the node automatically calculates optimal Stage 1 steps using one of three methods:

### Simple Math
**When**: `lightning_start=1` (default value)
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

**Quality Threshold**: Uses `base_quality_threshold` from config to ensure sufficient base sampling resolution.

### Switch Step Auto-Calculation
When you set `switch_step=-1` (Advanced node only), the node automatically calculates the switching point:
- Calculates 50% midpoint of lightning steps (rounded up)
- Common switching point, but may require experimentation based on your scheduler
- Provides balanced processing between lightning high and low models

**Log Messages**: You'll see method names in the console output when using auto-calculation, helping you understand which approach was used for your specific configuration.

## Edge Cases and Special Modes (Advanced Node Only)

The Advanced node supports several special sampling modes by configuring specific parameter combinations. Each mode has strict validation requirements:

### 1. Lightning-Only Mode (Skip Stage 1)
- **Configuration**: `lightning_start=0`
- **Requirements**: `base_steps` must be `-1` or `0` (validation enforced)
- **Behavior**: Skips base model entirely, starts with Lightning High
- **Use case**: Pure Lightning LoRA workflow

### 2. Base High + Lightning Low Mode (Skip Stage 2)
- **Configuration**: Set `lightning_start` equal to the switch point
- **Automatic**: Occurs when `lightning_start == switch_step`
- **Behavior**: Base high-noise processing → direct jump to Lightning Low
- **Use case**: Simplified two-stage workflow emphasizing noise transition

### 3. Lightning Low Only Mode (Skip Stages 1 & 2)
- **Configuration**:
  - `lightning_start=0`
  - `switching_strategy="Manual switch step"`
  - `switch_step=0`
- **Requirements**: `base_steps` must be `-1` or `0` (validation enforced)
- **Behavior**: Only Lightning Low refinement stage executes
- **Use case**: Final polish on pre-processed latents

### 4. Validation Rules
- `lightning_start > 0` requires `base_steps >= 1`
- `base_steps = 0` only allowed when `lightning_start = 0`
- When both stages 1&2 are skipped, `base_steps` must be `-1` or `0`
- `lightning_start` cannot exceed `switch_step`

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
- `nodes.py`: Main implementation with clean inheritance hierarchy:
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