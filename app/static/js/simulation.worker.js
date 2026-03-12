/**
 * PECS Simulation Worker v4.0 - Static Neural Generation
 * Replaces physics engine with deterministic mathematical layout
 * for instant rendering of massive datasets (20k-100k+ nodes).
 */


let simulation = null;
let simNodes = [];

self.onmessage = function(e) {
    const { type, nodes, links, config } = e.data;

    if (type === 'init') {
        simNodes = nodes;

        // 1. Build Adjacency Map for O(1) lookup
        const adj = new Map();
        nodes.forEach(n => adj.set(n.id, []));
        links.forEach(l => {
            if(adj.has(l.source_id)) adj.get(l.source_id).push(l.target_id);
            if(adj.has(l.target_id)) adj.get(l.target_id).push(l.source_id);
        });

        // 2. Sort by degree (descending) to place Hubs first
        // We create a sorted index to process nodes in order without breaking the original array order
        const sortedIndices = simNodes.map((n, i) => i).sort((a, b) => 
            (simNodes[b].degree || 0) - (simNodes[a].degree || 0)
        );

        const placed = new Set();
        const nodeMap = new Map(simNodes.map(n => [n.id, n]));
        
        // Scale spread based on node count to prevent congestion
        const baseSpread = config.distance || 120;
        const spread = simNodes.length > 2000 ? baseSpread * 12.0 : baseSpread;
        
        // 3. Mathematical Layout Generation
        sortedIndices.forEach((index, i) => {
            const node = simNodes[index];
            
            // Find neighbors that have already been placed
            const neighbors = adj.get(node.id) || [];
            const placedNeighbors = neighbors
                .map(id => nodeMap.get(id))
                .filter(n => placed.has(n.id));

            if (placedNeighbors.length > 0) {
                // CLUSTER STRATEGY: Place near connected neighbors
                let avgX = 0, avgY = 0;
                placedNeighbors.forEach(n => { avgX += n.x; avgY += n.y; });
                avgX /= placedNeighbors.length;
                avgY /= placedNeighbors.length;

                // CLUSTER BIAS: Push clusters away from the center to avoid middle congestion
                // Calculate angle from center (0,0) to the average position
                const angleFromCenter = Math.atan2(avgY, avgX);
                const pushOut = spread * 2.5;
                
                // Place node: Average of neighbors + Push outward + Organic Jitter
                node.x = avgX + Math.cos(angleFromCenter) * pushOut + (Math.random() - 0.5) * spread;
                node.y = avgY + Math.sin(angleFromCenter) * pushOut + (Math.random() - 0.5) * spread;
            } else {
                // SPIRAL STRATEGY: Place hubs/disconnected nodes on a Phyllotaxis spiral
                // Golden Angle = ~2.4 radians
                const angle = i * 2.4;
                // Power 0.6 spreads outer nodes more than sqrt (0.5), reducing center density
                const radius = spread * Math.pow(i + 10, 0.65) * 3.0; 
                
                node.x = Math.cos(angle) * radius;
                node.y = Math.sin(angle) * radius;
            }
            
            placed.add(node.id);
        });

        // Render once and stop
        sendPositions();
        self.postMessage({ type: 'end' });
    }

    if (type === 'reheat') {
        // Physics disabled - no reheat needed
    }

    if (type === 'fix') {
        // Manual drag handling without physics
        if (simNodes[e.data.index]) {
            simNodes[e.data.index].x = e.data.x;
            simNodes[e.data.index].y = e.data.y;
            sendPositions(); // Instant update
        }
    }

    if (type === 'unfix') {
        // No physics to release into
    }

    if (type === 'stop') {
        // Already stopped
    }

    if (type === 'updateConfig') {
        // Static layout doesn't update dynamically
    }
};

function sendPositions() {
    const positions = new Float32Array(simNodes.length * 2);
    for (let i = 0; i < simNodes.length; i++) {
        positions[i * 2]     = simNodes[i].x || 0;
        positions[i * 2 + 1] = simNodes[i].y || 0;
    }
    self.postMessage({ type: 'tick', positions }, [positions.buffer]);
}