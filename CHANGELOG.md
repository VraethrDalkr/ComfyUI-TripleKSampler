# CHANGELOG

All notable changes to this project are documented in this file.

## [Unreleased]
- Future enhancement ideas: additional strategies, better boundary computation, performance optimizations, extended model compatibility

## [0.7.9] - 2025-09-18
### Improved
- Enhanced README.md with comprehensive parameter documentation and clearer organization
- Added "Key Differences from Native KSampler" section for better user understanding
- Improved node tooltips for better parameter clarity and context
- Updated config documentation with clearer quality threshold explanation
- Reorganized documentation sections for logical flow and eliminated redundancy

## [0.7.8] - 2025-09-17
### Fixed
- Complete GitHub Actions workflow setup and testing

## [0.7.7] - 2025-09-17
### Fixed
- GitHub Actions publishing workflow testing

## [0.7.6] - 2025-09-17
### Added
- GitHub Actions workflow for automated ComfyUI Registry publishing

## [0.7.5] - 2025-09-17
### Fixed
- Remove [build-system] section from pyproject.toml to improve ComfyUI Registry compatibility

## [0.7.4] - 2024-09-17
### Fixed
- Update CHANGELOG.md for v0.7.3 that was missed in previous release
- Enhanced CLAUDE.md with comprehensive version handling instructions
- Verified version references are limited to pyproject.toml, __init__.py, and CHANGELOG.md

## [0.7.3] - 2024-09-17
### Fixed
- Remove conflicting fields from pyproject.toml for ComfyUI-Manager compatibility
- Removed 'classifiers' and 'requires-comfyui' fields causing installation conflicts

## [0.7.2] - 2024-09-17
### Changed
- Repository made public for ComfyUI Registry and community access
- Added node.zip to .gitignore for Comfy CLI compatibility

## [0.7.1] - 2024-09-17
### Added
- Minimal 8x8 latent optimization for dry run mode to speed up downstream VAE processing
- ComfyUI Registry configuration in pyproject.toml for ComfyUI-Manager discovery
- CHANGELOG.md for better version tracking separated from CLAUDE.md

## [0.7.0] - 2024-09-17
### Added
- Main module renamed from `triple_ksampler_wan22.py` to `nodes.py` following ComfyUI convention
- Example workflows reorganized from root directory to `example_workflows/` with cleaner naming
- Added `lightning_start` parameter to Simple node for increased flexibility
- Improved test suite from 80 to 87 passing tests (100% success rate)
- Added comprehensive VSCode type checking configuration via `pyrightconfig.json`
### Changed
- Documentation updated for new file locations and structure

## [0.6.1] - 2024-09-16
### Added
- `lightning_start` parameter to Simple node for increased flexibility
- Reorganized example workflows into `example_workflows/` directory with cleaner naming
- Renamed main module from `triple_ksampler_wan22.py` to `nodes.py` following ComfyUI convention
### Changed
- Documentation updated for new file locations and structure

## [0.6.0] - 2024-09-14
### Changed
- **BREAKING**: Configuration parameter renamed from `min_total_steps` to `base_quality_threshold` for clarity
- Improved parameter naming - now clearly indicates it's a quality threshold for base model auto-calculation
### Updated
- All config examples and descriptions to use clearer terminology

## [0.5.0] - 2024-09-11
### Changed
- **BREAKING**: Configuration format changed from JSON to TOML (`config.toml` instead of `config.json`)
- Cleaner config format with comment support and no confusion with ComfyUI workflows
### Added
- `requirements.txt` for proper TOML dependency management (`tomli` for Python <3.11)
- Enhanced TOML loading with graceful fallbacks and better error handling
### Removed
- Detailed instructions from example workflows documentation

## [0.4.0] - 2024-09-10
### Fixed
- Division by zero error in overlap detection for edge cases with `base_steps=0`
### Changed
- Improved logging consistency - step calculation logging now appears before model switching logging for both auto and manual `base_steps`
- Consolidated all step calculations to happen together before model switching strategy calculation

## [0.3.2] - 2024-09-09
### Removed
- **BREAKING**: KJNodes compatibility system entirely (79 lines of code eliminated)
### Added
- Dry run mode as UI parameter in advanced node for testing configurations
- Enhanced testing environment with full ComfyUI integration
### Fixed
- Improved type annotations for better IDE support (`stage_info` parameter)
### Changed
- Streamlined codebase with cleaner inheritance architecture

## [0.3.1] - 2024-09-07
### Changed
- **BREAKING**: Parameter names simplified for clarity (`switching_strategy` → `switch_strategy`, `midpoint` → `switch_step`, `boundary` → `switch_boundary`)
- **BREAKING**: Strategy option renamed ("50% of lightning steps" → "50% of steps")
- Simplified internal variable naming and stage execution logic
### Added
- Comprehensive edge case validation and error handling
- Clean visual separators with bare logger for empty lines
### Fixed
- Proper noise addition for Stage3-only scenarios
- Dynamic UI restricted to advanced node only

## [0.2.0] - 2024-09-05
### Changed
- **BREAKING**: Node names swapped for better UX
- **BREAKING**: Parameter `manual_midpoint` renamed to `midpoint`
- Clean Git Workflow: Release-only main branch established
### Added
- Enhanced Dropdown Strategy: 5 options for Advanced, 3 for Simple node
- Dynamic UI: JavaScript extension for real-time parameter visibility control
- TOML Configuration: User-configurable settings in `config.toml` to avoid git conflicts
- Auto-boundary Selection: T2V (0.875) and I2V (0.900) boundaries
- Quality Threshold: `MIN_TOTAL_STEPS` set to 20 for optimal balance

## [0.1.0] - 2024-09-01
### Added
- Initial release with configurable `_MIN_TOTAL_STEPS` constant
- Professional code structure and Apache 2.0 license

## [0.0.0] - 2024-08-30
### Added
- Initial development version