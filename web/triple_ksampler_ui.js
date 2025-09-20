import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Helper function to find widget by name
function findWidgetByName(node, name) {
    return node.widgets?.find((w) => w.name === name);
}

// Unified CSS visibility approach for all dynamic widgets
function toggleWidget(node, widgetName, show = false) {
    if (!node.widgets) return;

    const widget = node.widgets.find((w) => w.name === widgetName);
    if (!widget) return;

    // Use CSS to show/hide instead of manipulating widget array
    if (widget.element) {
        widget.element.style.display = show ? "" : "none";

        // Also hide the label if it exists
        if (widget.element.previousElementSibling &&
            widget.element.previousElementSibling.textContent === widgetName + ":") {
            widget.element.previousElementSibling.style.display = show ? "" : "none";
        }
    }

    // Mark widget as hidden/shown for ComfyUI's awareness
    widget.hidden = !show;
}



// Dynamic UI extension for TripleKSampler nodes
app.registerExtension({
    name: "TripleKSampler.DynamicUI",

    async nodeCreated(node) {
        // Only apply to the advanced TripleKSampler node
        if (node.comfyClass !== "TripleKSamplerWan22LightningAdvanced") {
            return;
        }

        // Find strategy widget
        const strategyWidget = findWidgetByName(node, "switch_strategy");
        if (!strategyWidget) return;

        // Function to update widget visibility
        const updateWidgetVisibility = (strategy) => {
            let showSwitchStep = false;
            let showSwitchBoundary = false;

            switch (strategy) {
                case "50% of steps":
                    showSwitchStep = false;
                    showSwitchBoundary = false;
                    break;
                case "Manual switch step":
                    showSwitchStep = true;
                    showSwitchBoundary = false;
                    break;
                case "T2V boundary":
                case "I2V boundary":
                    showSwitchStep = false;
                    showSwitchBoundary = false;
                    break;
                case "Manual boundary":
                    showSwitchStep = false;
                    showSwitchBoundary = true;
                    break;
                default:
                    showSwitchStep = true;
                    showSwitchBoundary = true;
            }

            // Apply visibility changes using unified CSS approach
            toggleWidget(node, "switch_step", showSwitchStep);
            toggleWidget(node, "switch_boundary", showSwitchBoundary);

            // Refresh canvas
            node.setDirtyCanvas(true, true);
            if (node.graph && node.graph.canvas) {
                node.graph.canvas.setDirty(true, true);
            }

            setTimeout(() => {
                if (node.onResize) {
                    node.onResize(node.size);
                }
                node.setDirtyCanvas(true, true);
            }, 10);
        };

        // Set up callback for strategy changes
        const originalCallback = strategyWidget.callback;
        strategyWidget.callback = function (value) {
            updateWidgetVisibility(value);
            if (originalCallback) {
                originalCallback.apply(this, arguments);
            }
        };

        // Apply initial visibility based on default value
        setTimeout(() => {
            updateWidgetVisibility(strategyWidget.value);
        }, 100);

        // Find base_steps widget for base_quality_threshold visibility
        const baseStepsWidget = findWidgetByName(node, "base_steps");
        if (baseStepsWidget) {
            // Function to update base_quality_threshold visibility
            const updateBaseQualityThresholdVisibility = (baseStepsValue) => {
                // Show base_quality_threshold only when base_steps is -1 (auto-calculation mode)
                const showBaseQualityThreshold = baseStepsValue === -1;
                toggleWidget(node, "base_quality_threshold", showBaseQualityThreshold);

                // Refresh canvas
                node.setDirtyCanvas(true, true);
                if (node.graph && node.graph.canvas) {
                    node.graph.canvas.setDirty(true, true);
                }

                setTimeout(() => {
                    if (node.onResize) {
                        node.onResize(node.size);
                    }
                    node.setDirtyCanvas(true, true);
                }, 10);
            };

            // Set up callback for base_steps changes
            const originalBaseStepsCallback = baseStepsWidget.callback;
            baseStepsWidget.callback = function (value) {
                updateBaseQualityThresholdVisibility(value);
                if (originalBaseStepsCallback) {
                    originalBaseStepsCallback.apply(this, arguments);
                }
            };

            // Apply initial visibility based on default value (CSS approach)
            setTimeout(() => {
                const baseStepsValue = baseStepsWidget.value;
                updateBaseQualityThresholdVisibility(baseStepsValue);
            }, 100);
        }

        // Hide the dry_run widget using the same approach as other dynamic widgets
        setTimeout(() => {
            toggleWidget(node, "dry_run", false);
        }, 50);

        // Add native dry run button widget at the bottom
        setTimeout(() => {
            const dryRunButton = node.addWidget("button", "🧪 Run Dry Run", null, () => {
                // Find and set the dry_run widget value to true
                const dryRunWidget = findWidgetByName(node, "dry_run");
                if (dryRunWidget) {
                    dryRunWidget.value = true;
                }
                // Queue execution immediately
                app.queuePrompt(0, 1);
                // Note: Auto-reset is handled by the "executed" event listener in setup()
            }, {
                serialize: false
            });


            // Move the hidden dry_run widget to appear right after the button for cleaner spacing
            setTimeout(() => {
                const dryRunWidget = findWidgetByName(node, "dry_run");
                if (dryRunWidget && node.widgets) {
                    // Remove dry_run widget from its current position
                    const dryRunIndex = node.widgets.indexOf(dryRunWidget);
                    if (dryRunIndex > -1) {
                        node.widgets.splice(dryRunIndex, 1);
                        // Add it back at the end (right after the button)
                        node.widgets.push(dryRunWidget);
                    }
                }
            }, 50);

            // Ensure button stays at bottom and refresh canvas
            node.setDirtyCanvas(true, true);
            if (node.graph && node.graph.canvas) {
                node.graph.canvas.setDirty(true, true);
            }
        }, 200); // Slight delay to ensure all other widgets are ready

    },

    // Listen for overlap warnings from backend and execution events
    setup() {
        api.addEventListener("triple_ksampler_overlap", (ev) => {
            try {
                const payload = ev.detail;
                if (!payload) return;

                app.extensionManager.toast.add({
                    severity: payload.severity || "warn",
                    summary: payload.summary || "TripleKSampler",
                    detail: payload.detail || "",
                    life: payload.life || 5000,
                });
            } catch (e) {
                console.error(
                    "TripleKSampler extension failed to handle overlap message:",
                    e
                );
            }
        });

        // Listen for dry run completion notifications
        api.addEventListener("triple_ksampler_dry_run", (ev) => {
            try {
                const payload = ev.detail;
                if (!payload) return;

                app.extensionManager.toast.add({
                    severity: payload.severity || "info",
                    summary: payload.summary || "TripleKSampler: Dry Run",
                    detail: payload.detail || "",
                    life: payload.life || 12000,
                });
            } catch (e) {
                console.error(
                    "TripleKSampler extension failed to handle dry run message:",
                    e
                );
            }
        });

        // Listen for execution completion to reset dry_run parameters
        api.addEventListener("executed", (ev) => {
            try {
                // Reset dry_run to false for all TripleKSampler nodes after execution
                if (app.graph && app.graph._nodes) {
                    for (const node of app.graph._nodes) {
                        if (node.comfyClass === "TripleKSamplerWan22LightningAdvanced") {
                            const dryRunWidget = findWidgetByName(node, "dry_run");
                            if (dryRunWidget && dryRunWidget.value === true) {
                                dryRunWidget.value = false;
                            }
                        }
                    }
                }
            } catch (e) {
                console.error(
                    "TripleKSampler extension failed to handle execution event:",
                    e
                );
            }
        });
    },
});
