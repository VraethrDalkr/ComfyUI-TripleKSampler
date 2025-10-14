# CHANGELOG

All notable changes to this project are documented in this file.

## [0.8.7] - 2025-10-14
### Fixed
- Node resize loop during image generation causing window shaking (issue #5)
- Removed aggressive `onResize()` calls from widget visibility update functions

## [0.8.6] - 2025-10-14
### Changed
- Dry run mode now interrupts workflow execution instead of returning minimal latent to prevent downstream nodes (VAE Decode) from processing
- Enhanced dry run completion message for clarity: "[DRY RUN] Complete - interrupting workflow execution (expected behavior)"
- JavaScript UI now listens for both "executed" and "execution_interrupted" events to properly reset dry_run widget state

### Removed
- Unused dry run context menu implementation (`run_dry_run()` method)
- Deprecated `_create_dry_run_minimal_latent()` method and MIN_LATENT constants (no longer needed)
- Unnecessary try/finally wrapper for dry run flag management (simplified code flow)

### Fixed
- Dry run widget state persistence issue where subsequent workflow executions would re-run dry run instead of normal sampling
- Tooltip updated to reflect actual behavior: "test stage calculations without actual sampling"

## [0.8.5] - 2025-10-14
### Fixed
- Graceful handling of user cancellations: InterruptProcessingException now propagates correctly without wrapping, preventing misleading error dialogs when users cancel sampling operations

## [0.8.4] - 2025-10-14
### Added
- Switch Strategy utility nodes for external strategy control (Simple and Advanced variants)
- Dynamic UI support for automatic widget visibility when strategy nodes are connected
- Hybrid workflow example demonstrating T2V/I2V with different strategies

### Changed
- Display names renamed to "TripleKSampler (Simple)" and "TripleKSampler (Advanced)" for clarity
- Documentation improvements for clarity and consistency

## [0.8.3] - 2025-09-22
### Changed
- Cleaned up CHANGELOG.md to remove internal development references for public release readiness
- Streamlined .gitignore to remove redundant entries and development artifacts

## [0.8.2] - 2025-09-22
### Added
- Model download guidance notes in all 3 example workflows
- Custom LoRA workflow example (t2v_custom_lora_workflow.json) demonstrating layered LoRA usage
- Comprehensive download links to official Comfy-Org repackaged models
- Custom LoRA workflow guidance in README for both T2V and I2V usage

### Changed
- README simplified from 385 to 66 lines with GitHub wiki links for detailed documentation
- Wiki content organization and structure improvements
- Error message formatting for cleaner display (removes extra colon from empty exceptions)
- Parameter documentation reordered to match UI layout
- Lightning parameter descriptions clarified for lightning stages specifically

### Fixed
- Wiki duplicate titles removed to prevent GitHub auto-title conflicts
- Parameter Reference updated with proper file placement paths
- Troubleshooting section updated with quantized model guidance and sigma_shift range
- Quality Threshold description clarified for Advanced vs Simple node usage

### Documentation
- Complete wiki restructure with 8 organized pages
- Model download instructions with proper file placement
- Lightning LoRA sources and custom LoRA integration examples
- Improved troubleshooting with GGUF quantization guidance

## [0.8.1] - 2025-09-21
### Added
- Comprehensive inline code documentation for better understanding
- Enhanced error messages with detailed exception information for easier debugging
- Improved testing compatibility for more stable operation across environments

### Changed
- README documentation significantly improved with clearer parameter explanations
- Better organization of parameter reference and usage instructions
- Enhanced auto-calculation method documentation for user clarity

### Fixed
- Conditional server imports to prevent crashes in testing environments
- Improved error handling with exception type information for better troubleshooting

## [0.8.0] - 2025-09-19
### Added
- `base_quality_threshold` parameter exposed in Advanced node UI for runtime configuration
- Dynamic widget visibility: parameter only shows when `base_steps = -1` (auto-calculation mode)
- Enhanced dry run mode with toast notifications showing calculated values
- Comprehensive test coverage for experimental UI features

### Changed
- **BREAKING**: Parameter reordering in Advanced node requires workflow recreation
- `dry_run` parameter moved to end of required parameters for better organization
- `base_quality_threshold` moved from required to optional section to fix ComfyUI validation
- Parameter range changed from `-1 to 100` to `1 to 100` with config.toml default (20)
- Updated example workflows to reflect new parameter layout

### Fixed
- Widget value preservation during dynamic visibility changes
- Parameter validation for new base_quality_threshold range
- Test suite updated for breaking changes

## [0.7.11] - 2025-09-18
### Added
- "Why TripleKSampler vs Multiple KSamplers?" documentation section explaining step resolution philosophy
- Clear comparison between typical multi-KSampler setups and TripleKSampler approach
- Design philosophy explanation for step resolution vs denoising percentage separation

### Improved
- Refined documentation to reduce redundancy while preserving functionality information
- Added cross-references to detailed technical sections

## [0.7.10] - 2025-09-18
### Added
- Edge Cases and Special Modes documentation section in README.md
- Comprehensive documentation for Lightning-Only Mode, Base High + Lightning Low Mode, and Lightning Low Only Mode
- Complete validation rules and parameter requirements for special sampling configurations

## [0.7.9] - 2025-09-18
### Improved
- Enhanced README.md with comprehensive parameter documentation and clearer organization
- Added "Key Differences from Native KSampler" section for better user understanding
- Improved node tooltips for better parameter clarity and context
- Updated config documentation with clearer quality threshold explanation
- Reorganized documentation sections for logical flow and eliminated redundancy

## [0.7.8] - 2025-09-17
### Fixed
- ComfyUI Registry publishing compatibility

## [0.7.7] - 2025-09-17
### Fixed
- Registry publishing workflow compatibility

## [0.7.6] - 2025-09-17
### Added
- Automated ComfyUI Registry publishing support

## [0.7.5] - 2025-09-17
### Fixed
- Remove [build-system] section from pyproject.toml to improve ComfyUI Registry compatibility

## [0.7.4] - 2024-09-17
### Fixed
- Update CHANGELOG.md for v0.7.3 that was missed in previous release

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
- CHANGELOG.md for better version tracking

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