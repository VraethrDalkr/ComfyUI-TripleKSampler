# ComfyUI-TripleKSampler

Professional triple-stage sampling nodes for Wan2.2 split models with Lightning LoRA integration. Simple to use, powerful results.

## Features

- **Triple-Stage Workflow** - Base denoising → Lightning high → Lightning low
- **Two Node Variants** - Simple (smart defaults) and Advanced (full control)
- **Intelligent Auto-Calculation** - Optimal parameter computation
- **Model-Safe Cloning** - No mutation of original models
- **Sigma Shift Integration** - Built-in ModelSamplingSD3 support

## Quick Start

1. **Install**
   ```bash
   cd ComfyUI/custom_nodes/
   git clone https://github.com/VraethrDalkr/ComfyUI-TripleKSampler.git
   cd ComfyUI-TripleKSampler && pip install -r requirements.txt
   ```

2. **Use** - Find nodes under `TripleKSampler/sampling` category after ComfyUI restart

3. **Configure** - Connect your Wan2.2 models and set basic parameters

## Node Types

| Node | Best For | Key Features |
|------|----------|--------------|
| **TripleKSampler** | Most users | Smart defaults, auto-calculation, streamlined interface |
| **TripleKSampler Advanced** | Power users | Full control, 5 switching strategies, dry-run testing |

## Essential Parameters

- **sigma_shift** - Sigma shift value (default: 5.0)
- **base_cfg** - CFG for base denoising (default: 3.5)
- **lightning_steps** - Total lightning steps (default: 8)
- **lightning_start** - Starting step in lightning schedule (default: 1)

## Documentation

- **[📖 Complete Documentation](../../wiki)** - Comprehensive guides and reference
- **[⚙️ Installation Guide](../../wiki/Installation-Guide)** - Detailed setup instructions
- **[📋 Parameter Reference](../../wiki/Parameter-Reference)** - Full parameter documentation
- **[🔧 Configuration Guide](../../wiki/Configuration-Guide)** - TOML configuration setup
- **[🎯 Model Switching Strategies](../../wiki/Model-Switching-Strategies)** - Strategy explanations
- **[🚀 Advanced Features](../../wiki/Advanced-Features)** - Edge cases and special modes
- **[🛠️ Troubleshooting](../../wiki/Troubleshooting)** - Common issues and solutions

## Example Workflows

Example workflows are included in the `example_workflows/` directory.

## Support

- **Issues** - [GitHub Issues](https://github.com/VraethrDalkr/ComfyUI-TripleKSampler/issues)
- **Documentation** - [Project Wiki](../../wiki)
- **Updates** - [Changelog](CHANGELOG.md)

## License

Apache 2.0 License - see [LICENSE](LICENSE) file for details.

---

**Author**: [VraethrDalkr](https://github.com/VraethrDalkr)