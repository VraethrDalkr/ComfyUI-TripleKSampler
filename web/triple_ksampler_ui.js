import { app } from "../../scripts/app.js";

// Helper function to find widget by name
function findWidgetByName(node, name) {
    return node.widgets?.find(w => w.name === name);
}

// Helper function to toggle widget visibility
function toggleWidget(node, widget, show = false) {
    if (!widget) return;
    
    widget.type = show ? widget.origType || widget.type : "hidden";
    widget.computeSize = show ? 
        (widget.origComputeSize || LiteGraph.LGraphNode.prototype.computeSize) : 
        () => [0, -4];
    
    if (show && !widget.origType) {
        widget.origType = widget.type;
        widget.origComputeSize = widget.computeSize;
    }
}

// Dynamic UI extension for TripleKSampler nodes
app.registerExtension({
    name: "TripleKSampler.DynamicUI",
    
    async nodeCreated(node) {
        // Only apply to our TripleKSampler nodes
        if (!["TripleKSamplerWan22LightningAdvanced", "TripleKSamplerWan22Lightning"].includes(node.comfyClass)) {
            return;
        }

        // Find widgets
        const strategyWidget = findWidgetByName(node, "midpoint_strategy");
        const midpointWidget = findWidgetByName(node, "midpoint");
        const boundaryWidget = findWidgetByName(node, "boundary");
        
        if (!strategyWidget) return;

        // Store original widget types for restoration
        if (midpointWidget && !midpointWidget.origType) {
            midpointWidget.origType = midpointWidget.type;
            midpointWidget.origComputeSize = midpointWidget.computeSize;
        }
        if (boundaryWidget && !boundaryWidget.origType) {
            boundaryWidget.origType = boundaryWidget.type;
            boundaryWidget.origComputeSize = boundaryWidget.computeSize;
        }

        // Function to update widget visibility
        const updateWidgetVisibility = (strategy) => {
            let showMidpoint = false;
            let showBoundary = false;

            switch (strategy) {
                case "50% midpoint":
                    // Hide both parameters
                    showMidpoint = false;
                    showBoundary = false;
                    break;
                case "Manual midpoint":
                    // Show midpoint, hide boundary
                    showMidpoint = true;
                    showBoundary = false;
                    break;
                case "T2V boundary":
                case "I2V boundary":
                    // Hide both (auto-select boundary values)
                    showMidpoint = false;
                    showBoundary = false;
                    break;
                case "Manual boundary":
                    // Show boundary, hide midpoint
                    showMidpoint = false;
                    showBoundary = true;
                    break;
                default:
                    // Default: show both for safety
                    showMidpoint = true;
                    showBoundary = true;
            }

            // Apply visibility changes
            toggleWidget(node, midpointWidget, showMidpoint);
            toggleWidget(node, boundaryWidget, showBoundary);

            // Update node size and force redraw
            node.setSize(node.computeSize());
            node.setDirtyCanvas(true, true);
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
        updateWidgetVisibility(strategyWidget.value);
    }
});