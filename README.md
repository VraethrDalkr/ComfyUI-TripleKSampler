# ComfyUI-TripleKSampler

Triple-stage sampling nodes for Wan2.2 split models with Lightning LoRA integration.

## Features

- **Triple-Stage Workflow** - Base denoising → Lightning high → Lightning low
- **Six Node Variants** - Simple/Advanced/Advanced Alt for both native KSampler and WanVideoWrapper workflows
- **Intelligent Auto-Calculation** - Optimal parameter computation
- **Model-Safe Cloning** - No mutation of original models
- **Sigma Shift Integration** - Built-in ModelSamplingSD3 support
- **Automatic Sigma Refinement** - Theoretical optimization for perfect boundary alignment (refined strategies)

## Quick Start

1. **Install**
   ```bash
   cd ComfyUI/custom_nodes/
   git clone https://github.com/VraethrDalkr/ComfyUI-TripleKSampler.git
   cd ComfyUI-TripleKSampler && pip install -r requirements.txt
   ```

2. **Optional: WanVideoWrapper Integration** - Install [ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper) to enable TripleWVSampler nodes

3. **Use** - Find nodes under `TripleKSampler` category after ComfyUI restart
   - `TripleKSampler/sampling` - Native KSampler workflow nodes
   - `TripleKSampler/wanvideo` - WanVideoWrapper integration nodes
   - `TripleKSampler/utilities` - Switch Strategy utility nodes

4. **Configure** - Connect your Wan2.2 models and set basic parameters

## Why Use TripleKSampler?

The TripleKSampler node streamlines complex multi-model workflows while respecting base model step resolution. The diagram below compares four different approaches:

![Workflow Comparison](assets/workflows_compare.svg)

**Workflow Comparison:**
1. **Base Models Only** - Maximum quality, slowest generation (full base model processing)
2. **Lightning Models Only** - Minimum quality, fastest generation (full lightning processing)
3. **Typical 3 KSamplers** - Manual setup with decent quality and improved motion, but doesn't respect base model step resolution
4. **TripleKSampler Node** - Automated approach with decent quality, improved motion, and proper base model step resolution

The example shown uses `lightning_start=2`, `lightning_steps=8` with the default Base Quality Threshold and the 50% switch strategy. This demonstrates how TripleKSampler automates the complex model switching that would otherwise require manual KSampler coordination.

## Node Types

| Node | Category | Best For | Key Features |
|------|----------|----------|--------------|
| **TripleKSampler (Simple)** | sampling | Most users | Smart defaults, auto-calculation, streamlined interface |
| **TripleKSampler (Advanced)** | sampling | Power users | Full control, 8 switching strategies, dynamic UI, dry-run testing |
| **TripleKSampler (Advanced Alt)** | sampling | Power users | Full control, 8 switching strategies, static UI, dry-run testing - use if dynamic UI causes issues |
| **TripleWVSampler (Simple)** | wanvideo | WanVideoWrapper users | Smart defaults for TripleWVSampler workflows |
| **TripleWVSampler (Advanced)** | wanvideo | WanVideoWrapper power users | Full control for TripleWVSampler workflows, dynamic UI, dry-run testing |
| **TripleWVSampler (Advanced Alt)** | wanvideo | WanVideoWrapper power users | Full control for TripleWVSampler workflows, static UI, dry-run testing |
| **Switch Strategy (Simple)** | utilities | Simple node users | External strategy control, 5 strategies |
| **Switch Strategy (Advanced)** | utilities | Advanced node users | External strategy control, 8 strategies |

## Essential Parameters

- **sigma_shift** - Sigma shift value (default: 5.0)
- **base_cfg** - CFG for base denoising (default: 3.5)
- **lightning_start** - Starting step in lightning schedule (default: 1)
- **lightning_steps** - Total lightning steps (default: 8)

## Documentation

- **[📖 Complete Documentation](https://github.com/VraethrDalkr/ComfyUI-TripleKSampler/wiki)** - Comprehensive guides and reference
- **[⚙️ Installation Guide](https://github.com/VraethrDalkr/ComfyUI-TripleKSampler/wiki/Installation-Guide)** - Detailed setup instructions
- **[📋 Parameter Reference](https://github.com/VraethrDalkr/ComfyUI-TripleKSampler/wiki/Parameter-Reference)** - Full parameter documentation
- **[🔧 Configuration Guide](https://github.com/VraethrDalkr/ComfyUI-TripleKSampler/wiki/Configuration-Guide)** - TOML configuration setup
- **[🎯 Model Switching Strategies](https://github.com/VraethrDalkr/ComfyUI-TripleKSampler/wiki/Model-Switching-Strategies)** - Strategy explanations
- **[🚀 Advanced Features](https://github.com/VraethrDalkr/ComfyUI-TripleKSampler/wiki/Advanced-Features)** - Edge cases and special modes
- **[🛠️ Troubleshooting](https://github.com/VraethrDalkr/ComfyUI-TripleKSampler/wiki/Troubleshooting)** - Common issues and solutions

## Example Workflows

Example workflows are included in the `example_workflows/` directory.

**Text-to-Video (T2V)**:
- `t2v_simple.json` - Simple node with smart defaults
- `t2v_advanced.json` - Advanced node with full parameter control
- `t2v_simple_custom_lora.json` - Demonstrates layering custom LoRAs with Lightning LoRAs

**Image-to-Video (I2V)**:
- `i2v_simple.json` - Simple node with smart defaults
- `i2v_advanced.json` - Advanced node with full parameter control

**WanVideoWrapper Workflows** (requires [ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper)):
- `t2v_wanvideo_advanced.json` - Text-to-Video with TripleWVSampler Advanced
- `i2v_wanvideo_advanced.json` - Image-to-Video with TripleWVSampler Advanced

**Hybrid Workflow**: `hybrid_workflow.json` showcases the Switch Strategy utility nodes for external strategy control. Demonstrates using different switching strategies for T2V and I2V branches in a single workflow.
- **Requires**: [rgthree-comfy](https://github.com/rgthree/rgthree-comfy) custom nodes

**Math Node Comparison**: `tripleksampler_vs_math.json` demonstrates how to replicate TripleKSampler (Simple) behavior using manual math node calculations. This workflow provides a side-by-side comparison to help understand the internal calculations and validate the node's behavior.
- **Requires**: [ComfyUI-Easy-Use](https://github.com/yolain/ComfyUI-Easy-Use) and [ComfyUI-Custom-Scripts](https://github.com/pythongosssss/ComfyUI-Custom-Scripts)

## Known Limitations

### WanVideoWrapper Integration

**ComfyUI-WanVideoWrapper** is explicitly a work-in-progress project that receives frequent updates and integrates new features regularly. TripleWVSampler nodes:

- Cannot be comprehensively tested with all WanVideoWrapper features
- Some advanced features may not behave correctly with cascaded sampling
- Some features may conflict with Lightning LoRA workflows
- Some features may require specific denoising schedules incompatible with triple-stage sampling
- May break with WanVideoWrapper updates that change the sampler interface

**Before reporting issues with TripleWVSampler nodes:** Always test with the original WanVideoSampler node first to confirm the issue is specific to TripleWVSampler and not an upstream WanVideoWrapper issue.

## Support

- **Issues** - [GitHub Issues](https://github.com/VraethrDalkr/ComfyUI-TripleKSampler/issues)
- **Documentation** - [Project Wiki](https://github.com/VraethrDalkr/ComfyUI-TripleKSampler/wiki)
- **Updates** - [Changelog](CHANGELOG.md)

## License

Apache 2.0 License - see [LICENSE](LICENSE) file for details.

---

**Author**: [VraethrDalkr](https://github.com/VraethrDalkr)