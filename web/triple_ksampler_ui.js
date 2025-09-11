import { app } from "../../scripts/app.js";

// Dynamic UI extension for TripleKSampler nodes
app.registerExtension({
    name: "TripleKSampler.DynamicUI",
    
    async nodeCreated(node) {
        // Only apply to our TripleKSampler nodes
        if (!["TripleKSamplerWan22LightningAdvanced", "TripleKSamplerWan22Lightning"].includes(node.comfyClass)) {
            return;
        }

        // Find the midpoint_strategy widget
        const strategyWidget = node.widgets?.find(w => w.name === "midpoint_strategy");
        const midpointWidget = node.widgets?.find(w => w.name === "midpoint");
        const boundaryWidget = node.widgets?.find(w => w.name === "boundary");
        
        if (!strategyWidget) return;

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
                    // Default: show both
                    showMidpoint = true;
                    showBoundary = true;
            }

            // Apply visibility to widgets
            if (midpointWidget) {
                midpointWidget.type = showMidpoint ? "number" : "hidden";
                midpointWidget.computeSize = showMidpoint ? midpointWidget.computeSize : () => [0, -4];
            }
            
            if (boundaryWidget) {
                boundaryWidget.type = showBoundary ? "number" : "hidden";
                boundaryWidget.computeSize = showBoundary ? boundaryWidget.computeSize : () => [0, -4];
            }

            // Force node to redraw
            node.setDirtyCanvas(true, true);
        };

        // Set up callback for strategy changes
        strategyWidget.callback = (value) => {
            updateWidgetVisibility(value);
        };

        // Apply initial visibility based on default value
        updateWidgetVisibility(strategyWidget.value);
    }
});