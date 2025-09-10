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

Full-featured triple-stage sampler with complete parameter control:

- **Models**: Requires 3 models (high_model, high_model_lx2v, low_model_lx2v)
- **Sampling Control**: Configurable base_steps, lightning_start, lightning_steps
- **Model Switching**: Supports both midpoint and sigma boundary-based switching
- **CFG Control**: Separate CFG for base stage, Lightning stages use CFG=1.0

### Simple TripleKSampler (Wan2.2-Lightning)

Simplified variant with auto-computed parameters:

- **Simplified Interface**: Fewer exposed parameters
- **Auto-computed base_steps**: Ensures base_steps * lightning_steps >= 25
- **Fixed lightning_start**: Always set to 1
- **Same Quality**: Uses the same underlying algorithm

## Installation

1. Clone this repository to your ComfyUI custom_nodes directory:
   ```bash
   cd ComfyUI/custom_nodes/
   git clone https://github.com/VraethrDalkr/ComfyUI-TripleKSampler.git
   ```

2. Restart ComfyUI

3. The nodes will appear under `TripleKSampler/sampling` category

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
   - **Midpoint**: Simple 50/50 split between lightning models
   - **Boundary**: Sigma-based switching (0.875 for T2V, 0.900 for I2V)

### Parameter Guidelines

- **shift**: 5.0 is recommended for most cases
- **base_cfg**: 3.5 works well for most prompts
- **lightning_steps**: 8 steps provide good quality/speed balance
- **boundary**: 0.875 for text-to-video, 0.900 for image-to-video

## Model Switching Strategies

### Midpoint Strategy
Simple approach that switches at the midpoint of lightning steps. Reliable and straightforward.

### Sigma Boundary Strategy  
More sophisticated approach that analyzes sigma schedules to determine optimal switching point. Better quality but requires tuning boundary parameter for different use cases.

## Logging

The nodes provide detailed logging including:
- Stage execution with step ranges and denoising percentages
- Model switching strategy and computed switch points
- Auto-computed parameter values (Simple node)

## Development

### Requirements
- ComfyUI
- PyTorch
- Standard ComfyUI dependencies

### Code Structure
- `__init__.py`: Package initialization and node registration
- `triple_ksampler_wan22.py`: Main implementation with both node classes

### Testing
To test the nodes are properly loaded:
1. Restart ComfyUI
2. Check console for any import errors
3. Verify nodes appear in the node browser under `TripleKSampler/sampling`

## Version History

- **v0.1.0** - Initial release with configurable minimum total steps constant
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