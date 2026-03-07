/**
 * PECS Simulation Worker v3.0 - Organic Neural Physics
 * Prevents "Box" quadtree artifacts by scaling forces based on density
 * and applying a radial membrane force to shape the network.
 */

importScripts('https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js');

let simulation = null;
let simNodes = [];

self.onmessage = function(e) {
    const { type, nodes, links, config } = e.data;

    if (type === 'init') {
        simNodes = nodes;

        // Spawn nodes in a tight sphere instead of a massive galaxy
        simNodes.forEach(n => {
            if (!Number.isFinite(n.x) || !Number.isFinite(n.y)) {
                const r = Math.pow(Math.random(), 0.5) * 1500;
                const a = Math.random() * 2 * Math.PI;
                n.x = Math.cos(a) * r;
                n.y = Math.sin(a) * r;
            }
        });

        const nodeMap = new Map(simNodes.map((n, i) => [n.id, i]));
        const indexedLinks = links
            .map(l => ({
                source: nodeMap.get(l.source_id),
                target: nodeMap.get(l.target_id),
                source_id: l.source_id,
                target_id: l.target_id
            }))
            .filter(l => l.source !== undefined && l.target !== undefined);

        const nodeCount = simNodes.length;
        const isMassive = nodeCount > 2000;
        
        // Use config values if provided, otherwise fallback to stronger defaults
        const repulseStrength = config.repulsion || (isMassive ? -400 : -800);
        const linkDistance = config.distance || 100;
        
        if (simulation) simulation.stop();

        simulation = d3.forceSimulation(simNodes)
            .force("charge", d3.forceManyBody()
                .strength(repulseStrength)
                .distanceMax(isMassive ? 5000 : 10000)
                .theta(0.9)
            )
            .force("link", d3.forceLink(indexedLinks)
                .distance(d => {
                    // Organic variation: +/- 30% of target distance to break orbital rings
                    return linkDistance * (0.7 + Math.random() * 0.6);
                })
                .iterations(1)
            )
            .force("collide", d3.forceCollide()
                .radius(d => (d.degree ? Math.sqrt(d.degree) * 4 : 3) + 4)
                .strength(0.6)
                .iterations(1)
            )
            .force("x", d3.forceX(0).strength(isMassive ? 0.015 : 0.04))
            .force("y", d3.forceY(0).strength(isMassive ? 0.015 : 0.04))
            .alphaDecay(0.01)
            .velocityDecay(0.6) 
            .on('tick', sendPositions)
            .on('end', () => {
                sendPositions();
                self.postMessage({ type: 'end' });
            });
    }

    if (type === 'reheat') {
        if (!simulation) return;
        simulation.alpha(e.data.alpha || 0.3).restart();
    }

    if (type === 'fix') {
        if (simulation && simNodes[e.data.index]) {
            simNodes[e.data.index].fx = e.data.x;
            simNodes[e.data.index].fy = e.data.y;
        }
    }

    if (type === 'unfix') {
        if (simulation && simNodes[e.data.index]) {
            simNodes[e.data.index].fx = null;
            simNodes[e.data.index].fy = null;
        }
    }

    if (type === 'stop') {
        if (simulation) simulation.stop();
    }

    if (type === 'updateConfig') {
        if (!simulation) return;
        const { repulsion, distance } = config;
        if (repulsion !== undefined) simulation.force("charge").strength(repulsion);
        if (distance !== undefined) simulation.force("link").distance(d => {
            return distance * (0.7 + Math.random() * 0.6);
        });
        simulation.alpha(0.3).restart();
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