# CHANGELOG

All notable changes to this project are documented in this file.

## [Unreleased]

## [0.10.4] - 2025-11-14

### Changed
- **Debug logging improvements in Advanced nodes**
  - Removed redundant "Models:" line from TripleWVSampler debug output (always shows same model types)
  - Removed redundant "Loaded config defaults" line (base_quality_threshold already shown in parameters)
  - Added dry_run status to debug output for better visibility
  - Cleaner, more concise debug logging focusing on relevant information

## [0.10.3] - 2025-11-14

### Changed
- **Comprehensive documentation updates**
  - Updated README.md with TripleWVSampler node descriptions and refined strategy documentation
  - Updated wiki: Installation Guide, Parameter Reference, Model Switching Strategies, Advanced Features, Troubleshooting, Development Guide
  - Added WanVideoWrapper integration documentation
  - Clarified refined strategies as theoretical optimization (may not produce perceptible differences in most workflows)
  - Improved terminology consistency throughout documentation

### Fixed
- **Code documentation consistency**
  - Updated module docstrings to reflect current strategy counts (Simple: 5 strategies, Advanced: 8 strategies)
  - Fixed test file docstrings for new module structure (shared/ modules)
  - Updated inline comments to match refactored architecture

## [0.10.2] - 2025-11-14

### Fixed
- **Critical WanVideo registration bug breaking sample passing between stages (issue aec0e66)**
  - Root cause: Import-time verification in wvsampler_nodes.py (commit 2e51d5e) broke WanVideo initialization order
  - Symptom: Memory usage dropped from ~1.8GB (correct chained samples) to ~0.1GB (broken - fresh noise generated each stage)
  - Solution: Filesystem-gated registration in __init__.py with case-insensitive directory search
  - No import-time verification calls that trigger premature initialization
  - Evidence of fix: Stage 2 memory 1.235 GB ✓ (was 0.063 GB), Stage 3 memory 1.824 GB ✓ (was 0.079 GB)
  - Quality restored to baseline (was degraded)
- **Unit test isolation improved with structured model mocks**
  - All unit tests now use mock_model_factory fixture instead of bare MagicMock objects
  - Provides proper model structure (model.model.model_config) expected by ModelSamplingSD3
  - Reduces test pollution failures from 28 to 26 tests
  - All 204 unit tests pass when run separately, all 110 integration/regression tests pass

## [0.10.1] - 2025-11-11

### Added
- **Automatic sigma_shift refinement for boundary-based strategies (strategy-based activation)**
  - Adaptive bidirectional search algorithm starting from user's initial shift value (~20-50x faster than zero-start)
  - Perfect alignment between switch step and target boundary sigma
  - Supports all node variants: TripleKSampler, TripleKSamplerAdvanced, TripleWVSampler, TripleWVSamplerAdvanced
  - Works with both ComfyUI samplers (KSampler nodes) and WanVideo schedulers (WVSampler nodes)
  - Algorithm inspired by ComfyUI-WanMoEScheduler's iterative search approach (MIT License)
  - **New refined strategy variants** for better discoverability:
    - Advanced nodes: 8 strategies (add "T2V boundary (refined)", "I2V boundary (refined)", "Manual boundary (refined)")
    - Simple nodes: 5 strategies (add "T2V boundary (refined)", "I2V boundary (refined)")
    - Strategy utility nodes: Updated to match (SwitchStrategySimple: 5, SwitchStrategyAdvanced: 8)
    - Refined strategies auto-activate refinement (no config flag needed)
    - Toast notifications include refined shift in compact format: `(σ-shift: 5.00→6.94)`
  - Configuration options in `config.example.toml`:
    - `search_interval = 0.01` (precision control, smaller = more precise)
    - `tolerance = 0.001` (sigma matching threshold)
  - Info logging shows refinement: `"Refined sigma_shift: 5.0 → 6.94 for perfect alignment at step 4"`
  - Dry-run mode displays refinement in toast notification

### Changed
- **Sigma shift refinement activation changed from config-based to strategy-based**
  - Removed `enabled = false` flag from config.example.toml (no longer needed)
  - Refinement now activates automatically when refined strategy variant is selected
  - Better UX: refinement options visible in dropdown, per-node control, no hidden config flags
  - 100% backward compatible: refined strategies added at END of dropdown lists (existing workflow JSON indices unchanged)
- **Refined sigma shift display improved in toast notifications**
  - Refinement info now appears on separate indented line for better readability
  - Changed wording from "σ-shift: X→Y" to "σ-shift refined: X → Y" (clearer, better spacing)
- **Sigma shift refinement algorithm simplified and improved**
  - Removed user-facing config parameters (`search_interval`, `tolerance`) from config.example.toml
  - Algorithm now always finds the actual closest sigma match (removed premature tolerance early-exit checks)
  - Search interval hardcoded to optimal value (0.01 sigma units per iteration)
  - Configuration simplified: only essential user-facing settings remain in config file

### Changed
- **Improved WanVideo wrapper UX when ComfyUI-WanVideoWrapper not installed**
  - Reduced 6 repetitive warning messages to single clear INFO message at startup
  - Message now includes installation URL: `https://github.com/kijai/ComfyUI-WanVideoWrapper`
  - TripleWVSampler nodes only appear in UI when dependency is actually installed
  - Node tooltips simplified (dependency mention redundant when nodes visible)
- **Added consistent parameter tooltips to TripleWVSampler nodes**
  - Added `seed` and `sigma_shift` tooltips matching TripleKSampler conventions
  - Ensures consistent UX across all node variants

### Fixed
- **WVSampler strategy matching for refined variants**
  - Fixed TripleWVSampler nodes falling back to "50% of steps (fallback)" when using refined strategies
  - Strategy checking now correctly strips "(refined)" suffix before matching against base strategies
  - Affects: TripleWVSampler (Advanced) and TripleWVSamplerAdvanced nodes
- **Dynamic UI parameter visibility for refined strategies**
  - Fixed JavaScript UI not hiding/showing optional parameters correctly for refined strategy variants
  - Refined strategies now behave identically to their non-refined counterparts for parameter visibility
  - Example: "T2V boundary (refined)" now hides both switch_step and switch_boundary like "T2V boundary"
- **Fixed WVSampler parameter passing after conditional registration refactor**
  - Added `batched_cfg` and `rope_function` to `sample()` method signatures in both advanced.py and simple.py
  - Parameters were in required INPUT_TYPES but missing from method signatures after fallback removal
  - Ensures proper parameter flow: ComfyUI → sample() → WanVideoSampler.process()

## [0.10.0-dev] - 2025-11-10

### Changed
- **Symmetric architecture refactoring: Unified TripleKSampler and TripleWVSampler structure**
  - Renamed module hierarchy for clarity and consistency:
    - `triple_ksampler/core/` → `triple_ksampler/shared/` (truly shared utilities)
    - `triple_ksampler/nodes/` → `triple_ksampler/ksampler/` (TripleKSampler-specific)
    - Created `triple_ksampler/wvsampler/` for TripleWVSampler nodes
  - Extracted WanVideo wrapper into symmetric module structure:
    - `wvsampler/base.py` (~840 lines) - TripleWVSamplerAdvancedAlt
    - `wvsampler/advanced.py` (~30 lines) - Dynamic UI variant
    - `wvsampler/simple.py` (~280 lines) - Simplified interface
    - `wvsampler/utils.py` (~85 lines) - WanVideo lazy loader
  - Renamed root entry point files:
    - `nodes.py` → `ksampler_nodes.py` (TripleKSampler primary entry)
    - `nodes_wanvideo.py` → `wvsampler_nodes.py` (TripleWVSampler primary entry)
  - Both node types now follow identical structural patterns:
    - Shared utilities in `triple_ksampler/shared/`
    - Type-specific nodes in `ksampler/` and `wvsampler/`
    - Symmetric file organization (base.py, advanced.py, simple.py)
  - **INTERNAL RESTRUCTURING ONLY**: Zero breaking changes for users
    - ComfyUI NODE_CLASS_MAPPINGS keys unchanged
    - All node interfaces and behavior preserved
    - All 258 tests passing

- **Major internal refactoring: Improved code organization and maintainability (Phase 1-3)**
  - Reorganized node classes into focused modules (`triple_ksampler/ksampler/` package)
    - `base.py` (389 lines) - TripleKSamplerBase with core algorithm
    - `advanced.py` (557 lines) - Advanced node variants
    - `simple.py` (132 lines) - Simplified node
    - `strategy_nodes.py` (111 lines) - Strategy utility nodes
  - Extracted core logic to reusable modules (`triple_ksampler/shared/` package)
    - Created 7 focused modules: strategies, alignment, validation, config, logging, models, notifications
    - Added 156 comprehensive unit tests (100% coverage on critical functions)
  - Reduced file sizes significantly:
    - `ksampler_nodes.py`: 1168 → 114 lines (90% reduction, primary entry point)
    - `wvsampler_nodes.py`: 1327 → 110 lines (91% reduction, primary entry point)
  - Eliminated ~407 lines of code duplication (370 in nodes.py, 37 in nodes_wanvideo.py)
  - Improved SOLID compliance: Extracted 7/11 responsibilities from TripleKSamplerBase
  - Improved DRY compliance: 9/10 duplication items resolved

### Internal
- Symmetric architecture: Both node types (KSampler/WVSampler) follow identical patterns
- Clean naming: `shared/`, `ksampler/`, `wvsampler/` clearly indicate purpose
- Modernized test imports: Package imports instead of importlib patterns
- Each module under 400 lines (except wvsampler/base.py: ~840, advanced.py: 557 for multiple classes)
- All modules fully typed with modern Python 3.9+ syntax (`X | None`)
- Comprehensive docstrings on all public functions and classes

## [0.9.2] - 2025-10-15
### Added
- New TripleKSampler (Advanced Alt) node with static UI - all parameters always visible
- Stable alternative for users experiencing dynamic UI issues (issue #5)
- Node is immune to resize loops due to static parameter layout

### Changed
- Refactored class inheritance hierarchy for optimal DRY architecture (Base → AdvancedAlt → Advanced → Simple)
- Renamed internal class names for maintainability (TripleKSampler* from TripleKSamplerWan22*)
- Cleaned up __init__.py to minimal ComfyUI requirements
- Updated example workflows with stability notes pointing to Advanced Alt node

### Technical
- Full backward compatibility preserved via NODE_CLASS_MAPPINGS keys
- All 125 tests passing

## [0.9.1] - 2025-10-15
### Fixed
- Node resize loop during execution causing Advanced Sampler window to shake (issue #5)
- Implemented execution state tracking in JavaScript UI to prevent canvas updates during active execution
- Added deferred canvas update with 50ms safety delay after execution completes
- Consolidated update logic in handleExecutionComplete() helper function

## [0.9.0] - 2025-10-15
### Changed
- **BREAKING**: Advanced node now uses separate base/lightning sampler and scheduler parameters
  - New parameters: `base_sampler`, `base_scheduler`, `lightning_sampler`, `lightning_scheduler`
  - Removed parameters: `sampler_name`, `scheduler`
  - Enables independent control of sampling algorithms for base model vs lightning stages
  - Simple node maintains backward compatibility with original `sampler_name` and `scheduler` parameters

### Migration Guide
**Advanced Node Users**: Update workflows to specify:
- `base_sampler` + `base_scheduler` (for base model stages)
- `lightning_sampler` + `lightning_scheduler` (for lightning stages)

**Simple Node Users**: No changes required - continues using `sampler_name` and `scheduler`

## [0.8.8] - 2025-10-14
### Added
- Workflow comparison diagram (assets/workflows_compare.svg) visualizing 4 different sampling approaches
- "Why Use TripleKSampler?" section in README with embedded comparison diagram
- Math node comparison workflow documentation (TripleKSampler_vs_MathSetup.json)

### Changed
- Reorganized Example Workflows section with clearer progression from basic to advanced workflows
- Enhanced documentation for workflow dependencies and requirements

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