import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Helper function to find widget by name
function findWidgetByName(node, name) {
    return node.widgets?.find((w) => w.name === name);
}

// Helper function to actually remove/add widgets from node
function toggleWidget(node, widgetName, show = false) {
    if (!node.widgets) return;

    const existingIndex = node.widgets.findIndex((w) => w.name === widgetName);
    const widgetExists = existingIndex !== -1;

    if (show && !widgetExists) {
        // Need to add the widget back
        const widgetConfig = node.hiddenWidgets?.[widgetName];
        if (widgetConfig) {
            node.widgets.push(widgetConfig);
            delete node.hiddenWidgets[widgetName];
        }
    } else if (!show && widgetExists) {
        // Need to remove the widget
        const widget = node.widgets[existingIndex];

        // Store widget for later restoration
        if (!node.hiddenWidgets) node.hiddenWidgets = {};
        node.hiddenWidgets[widgetName] = widget;

        // Remove from widgets array
        node.widgets.splice(existingIndex, 1);
    }
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

            // Apply visibility changes
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
    },

    // Listen for overlap warnings from backend
    setup() {
        api.addEventListener("executed", (ev) => {
            try {
                const output = ev.detail?.output;
                if (!output) return;

                const payload = output.triple_ksampler_overlap;
                if (!payload) return;

                app.extensionManager.toast.add({
                    severity: payload.severity || "warn",
                    summary: payload.summary || "TripleKSampler",
                    detail: payload.detail || "",
                    life: payload.life || 5000,
                });
            } catch (e) {
                console.error(
                    "TripleKSampler extension failed to handle executed event:",
                    e
                );
            }
        });
    },
});
