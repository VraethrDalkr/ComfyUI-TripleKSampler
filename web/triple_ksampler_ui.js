import { app } from "../../scripts/app.js";

// Helper function to find widget by name
function findWidgetByName(node, name) {
    return node.widgets?.find(w => w.name === name);
}

// Helper function to actually remove/add widgets from node
function toggleWidget(node, widgetName, show = false) {
    if (!node.widgets) return;
    
    const existingIndex = node.widgets.findIndex(w => w.name === widgetName);
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
        // Only apply to the advanced TripleKSampler node (simple node has no dynamic parameters)
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
                    // Hide both parameters
                    showSwitchStep = false;
                    showSwitchBoundary = false;
                    break;
                case "Manual switch step":
                    // Show switch step, hide boundary
                    showSwitchStep = true;
                    showSwitchBoundary = false;
                    break;
                case "T2V boundary":
                case "I2V boundary":
                    // Hide both (auto-select boundary values)
                    showSwitchStep = false;
                    showSwitchBoundary = false;
                    break;
                case "Manual boundary":
                    // Show boundary, hide switch step
                    showSwitchStep = false;
                    showSwitchBoundary = true;
                    break;
                default:
                    // Default: show both for safety
                    showSwitchStep = true;
                    showSwitchBoundary = true;
            }

            // Apply visibility changes
            toggleWidget(node, "switch_step", showSwitchStep);
            toggleWidget(node, "switch_boundary", showSwitchBoundary);

            // Simple node layout refresh
            node.setDirtyCanvas(true, true);
            if (node.graph && node.graph.canvas) {
                node.graph.canvas.setDirty(true, true);
            }
            
            // Basic layout recalculation with minimal complexity
            setTimeout(() => {
                if (node.onResize) {
                    node.onResize(node.size);
                }
                node.setDirtyCanvas(true, true);
            }, 10);
        };

        // Set up callback for strategy changes
        const originalCallback = strategyWidget.callback;
        strategyWidget.callback = function(value) {
            updateWidgetVisibility(value);
            if (originalCallback) {
                originalCallback.apply(this, arguments);
            }
        };

        // Apply initial visibility based on default value
        setTimeout(() => {
            updateWidgetVisibility(strategyWidget.value);
        }, 100);
    }
});