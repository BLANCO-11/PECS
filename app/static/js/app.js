/**
 * PECS Neural Core - Organic Canvas Renderer
 * v3.6 - Biological Aesthetics Update
 */

const socket = io();

// Data & State
let nodes = [];
let links = [];
let simulation = null;
let canvas, ctx;
let width, height;
let renderingEnabled = true;

// D3 Zoom Behavior state
let transform = d3.zoomIdentity; 
let zoomBehavior = null; 

// Interaction State
let highlightedNode = null;
let selectedNode = null;
let linkedNeighbors = new Set();
let activationLevels = new Map();
let nodeMap = new Map(); // Optimization for lookups
let pulsePhase = 0;
let totalTokens = 0;

// Configuration - Deep Slate Theme
const CONFIG = {
    color: {
        bg: '#02020b', // Deep Blue/Dark Indigo Void
        node: {
            leaf: 'rgba(165, 180, 252, 0.4)', // Indigo-300
            hub: 'rgba(99, 102, 241, 0.9)',   // Indigo-500
            active: '234, 179, 8',           // Amber/Gold
            text: '226, 232, 240' // Changed to RGB base for dynamic opacity
        },
        edge: {
            leaf: 'rgba(99, 102, 241, 0.1)', 
            hub: 'rgba(99, 102, 241, 0.25)',  
            active: 'rgba(234, 179, 8, 0.6)', 
            highlight: 'rgba(129, 140, 248, 0.8)'
        }
    },
    physics: {
        repulsion: -600,
        distance: 120,
    }
};

// ==========================================
// 1. INITIALIZATION
// ==========================================

window.onload = () => {
    initCanvas();
    loadGraphData(); 
    setupUI();
};

function initCanvas() {
    const container = document.getElementById('graph-container');
    width = container.clientWidth;
    height = container.clientHeight;

    canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    container.appendChild(canvas);
    ctx = canvas.getContext('2d', { alpha: false });

    window.addEventListener('resize', resize);

    // Initial scale set to extremely zoomed out as requested
    const initialTransform = d3.zoomIdentity.translate(width / 2, height / 2).scale(0.015);

    zoomBehavior = d3.zoom()
        .scaleExtent([0.01, 4]) // Allowed to zoom out to 0.01 
        .on("zoom", e => { transform = e.transform; });

    d3.select(canvas)
        .call(zoomBehavior)
        .call(zoomBehavior.transform, initialTransform)
        .on("dblclick.zoom", null);

    const d3Canvas = d3.select(canvas);
    const originalWheel = d3Canvas.on("wheel.zoom"); 
    
    d3Canvas.on("wheel.zoom", function(event) {
        if (event.ctrlKey) {
            originalWheel.call(this, event);
            return;
        }
        event.preventDefault();
        const direction = event.deltaY < 0 ? 1 : -1;
        const multiplier = Math.pow(1.3, direction); 
        
        d3Canvas.transition().duration(300).ease(d3.easeCubicOut)
            .call(zoomBehavior.scaleBy, multiplier, d3.pointer(event));
    });

    d3.select(canvas)
        .call(d3.drag()
            .subject(dragSubject)
            .on("start", dragStarted)
            .on("drag", dragged)
            .on("end", dragEnded))
        .on("mousemove", handleMouseMove)
        .on("click", handleClick);
}

function resize() {
    const container = document.getElementById('graph-container');
    width = container.clientWidth;
    height = container.clientHeight;
    canvas.width = width;
    canvas.height = height;
}

// ==========================================
// 2. DATA LOADING & PHYSICS
// ==========================================

function loadGraphData() {
    fetch('/api/graph')
        .then(r => r.json())
        .then(data => {
            mergeGraphData(data);
        });
}

function mergeGraphData(data) {
    const currentNodesMap = new Map(nodes.map(n => [n.id, n]));
    const newNodes = [];

    data.nodes.forEach(n => {
        const displayName = n.object ? n.object : n.subject;

        if (currentNodesMap.has(n.id)) {
            const existing = currentNodesMap.get(n.id);
            existing.label = displayName;
            existing.raw = n;
            existing.degree = 0; // Reset for recalculation
            existing.links = []; // Reset links for adjacency
            newNodes.push(existing);
        } else {
            newNodes.push({
                id: n.id,
                label: displayName,
                raw: n,
                x: (Math.random() - 0.5) * 1000, 
                y: (Math.random() - 0.5) * 1000,
                degree: 0,
                // OPTIMIZATION: Initialize flags to avoid undefined checks in render
                _isLeaf: false,
                _isHub: false,
                links: [] // Adjacency list
            });
        }
    });

    const nodeLookup = new Map(newNodes.map(n => [n.id, n]));
    nodeMap = nodeLookup; // Update global map
    const newLinks = [];

    data.edges.forEach(e => {
        const sourceNode = nodeLookup.get(String(e.source_id));
        const targetNode = nodeLookup.get(String(e.target_id));

        if (sourceNode && targetNode) {
            const baseOffset = ((parseInt(e.source_id) + parseInt(e.target_id)) % 100) - 50;
            
            const link = {
                source: sourceNode,
                target: targetNode,
                type: e.type,
                curveOffset: baseOffset * 4.0, // Increased curvature
                // OPTIMIZATION: Store type flags here
                _isContra: e.type === 'contradicts'
            };

            newLinks.push(link);
            
            // Populate adjacency list for O(1) lookup
            sourceNode.links.push(link);
            targetNode.links.push(link);

            sourceNode.degree++;
            targetNode.degree++;
        }
    });

    // --- OPTIMIZATION: PRE-CALCULATE VISUAL FLAGS ONCE ---
    // Do this here instead of inside the render loop
    newNodes.forEach(n => {
        n._isLeaf = n.degree <= 1;
        n._isHub = n.degree > 5;
    });

    newLinks.forEach(l => {
        // Cache leaf status on the link itself
        l._isLeafLink = l.source._isLeaf || l.target._isLeaf;
        l._isHubLink = l.source._isHub || l.target._isHub;
        // OPTIMIZATION: Calculate importance for rendering priority
        l._importance = (l.source.degree || 0) + (l.target.degree || 0);
    });

    // OPTIMIZATION: Sort links by importance so we render the structural ones first
    newLinks.sort((a, b) => b._importance - a._importance);

    nodes = newNodes;
    links = newLinks;

    updateClusterNav(); 

    if (!simulation) {
        simulation = d3.forceSimulation(nodes)
            .force("charge", d3.forceManyBody().strength(d => d.degree <= 1 ? -100 : CONFIG.physics.repulsion))
            .force("link", d3.forceLink(links).id(d => d.id).distance(CONFIG.physics.distance))
            .force("center", d3.forceCenter(0, 0))
            .force("collide", d3.forceCollide(d => 8 + Math.sqrt(d.degree)))
            .stop(); 
            
        for (let i = 0; i < 40; ++i) simulation.tick();
        
        if (renderingEnabled) simulation.restart();
    } else {
        simulation.nodes(nodes);
        simulation.force("link").links(links);
        if (renderingEnabled) simulation.alpha(0.3).restart(); 
    }

    updateStatus(`${nodes.length} Nodes Active`);
    if(!pulsePhase) startAnimationLoop();
}


// ==========================================
// 3. UI LOGIC (CLUSTERS & ZOOM)
// ==========================================

function updateClusterNav() {
    const list = document.getElementById('cluster-list');
    const subjectMap = new Map();
    
    for (const n of nodes) {
        if (n.degree > 2) {
            const subj = n.raw.subject;
            if (!subjectMap.has(subj) || subjectMap.get(subj).degree < n.degree) {
                subjectMap.set(subj, n);
            }
        }
    }
    
    const hubs = Array.from(subjectMap.values())
                      .sort((a, b) => b.degree - a.degree)
                      .slice(0, 15);
    
    if(hubs.length === 0) {
        list.innerHTML = `<div style="padding: 12px; text-align: center; color: #64748b; font-size: 0.7rem;">No major clusters yet</div>`;
        return;
    }

    list.innerHTML = '';
    hubs.forEach(node => {
        const item = document.createElement('div');
        item.style.cssText = "padding: 8px 16px; border-bottom: 1px solid rgba(255,255,255,0.05); cursor: pointer; display: flex; justify-content: space-between; align-items: center;";
        const displayName = node.raw.subject.length > 20 ? node.raw.subject.substring(0,18)+'...' : node.raw.subject;
        item.innerHTML = `
            <span style="font-size: 0.75rem; color: #cbd5e1; text-transform: capitalize;">${displayName}</span>
            <span style="font-size: 0.65rem; color: #64748b; background: rgba(0,0,0,0.3); padding: 2px 6px; border-radius: 4px;">${node.degree}</span>
        `;
        item.onmouseenter = () => { item.style.background = 'rgba(255,255,255,0.05)'; };
        item.onmouseleave = () => { item.style.background = 'transparent'; };
        item.onclick = () => focusOnNode(node);
        list.appendChild(item);
    });
}

function focusOnNode(node) {
    if(!node) return;
    const scale = 0.6; 
    const x = -node.x * scale + width / 2;
    const y = -node.y * scale + height / 2;
    const targetTransform = d3.zoomIdentity.translate(x, y).scale(scale);

    d3.select(canvas).transition()
        .duration(1500)
        .ease(d3.easeCubicInOut)
        .call(zoomBehavior.transform, targetTransform);
        
    highlightedNode = node;
    openInspector(node);
}

// ==========================================
// 4. RENDER LOOP (ULTRA OPTIMIZED)
// ==========================================

function startAnimationLoop() {
    const loop = () => {
        if (!renderingEnabled) return requestAnimationFrame(loop);
        pulsePhase += 0.05;
        if(activationLevels.size > 0) {
            for(let [id, val] of activationLevels) {
                // OPTIMIZATION: Faster decay for non-hubs (leaves fade out quickly)
                const node = nodeMap.get(id);
                val *= (node && node._isHub) ? 0.98 : 0.7; 
                if(val < 0.01) activationLevels.delete(id);
                else activationLevels.set(id, val);
            }
        }
        render();
        requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
}

function render() {
    if (!renderingEnabled) return;
    ctx.fillStyle = CONFIG.color.bg;
    ctx.fillRect(0, 0, width, height);
    
    ctx.save();
    ctx.translate(transform.x, transform.y);
    ctx.scale(transform.k, transform.k);

    const pulse = (Math.sin(pulsePhase) + 1) / 2;

    const margin = 200 / transform.k; 
    const minX = -transform.x / transform.k - margin;
    const maxX = (width - transform.x) / transform.k + margin;
    const minY = -transform.y / transform.k - margin;
    const maxY = (height - transform.y) / transform.k + margin;

    // OPTIMIZATION: Strict bounds for edge culling (no margin)
    const strictMinX = -transform.x / transform.k;
    const strictMaxX = (width - transform.x) / transform.k;
    const strictMinY = -transform.y / transform.k;
    const strictMaxY = (height - transform.y) / transform.k;

    const hideLeaves = transform.k < 0.15;
    const extremeZoom = transform.k < 0.05;
    const ultraZoom = transform.k < 0.03; 
    const highDetailZoom = transform.k > 0.4; // Show all local details when zoomed in
    const useSpatialIndex = transform.k > 0.25; // OPTIMIZATION: Use node-first rendering when zoomed in
    // const simplifyLines = transform.k < 0.08; // Removed for cleaner rendering

    // --- PERFORMANCE LIMITS ---
    const MAX_VISIBLE_EDGES = 600; // Reduced to improve performance
    const MAX_HOVER_EDGES = 100;    // Reduced limit for hover context
    const MAX_ZOOMED_EDGES = 500;  // Reduced to prevent freeze on dense clusters
    const MAX_PARTIAL_EDGES = 50;   // Drastically reduced to prevent "star" explosion
    const MAX_ACTIVE_EDGES = 200;   // Strict limit for active edges to prevent freeze
    const MAX_ACTIVE_STRANDS = 100; // Limit for active edges going off-screen

    // --- BUCKETS FOR BATCH RENDERING ---
    const leafLinks = [];
    const hubLinks = [];
    const contraLinks = []; 
    const activeLinks = [];

    let partialEdgeCount = 0;
    let totalRenderedEdges = 0;
    let activeStrandCount = 0;

    // --- STRATEGY SELECTION ---
    let renderLinks = links;
    let renderNodes = nodes;

    if (useSpatialIndex) {
        // Node-First Strategy: Find visible nodes -> Get their edges
        renderNodes = [];
        const visibleLinksSet = new Set();

        for (const node of nodes) {
            if (node.x >= minX && node.x <= maxX && node.y >= minY && node.y <= maxY) {
                renderNodes.push(node);
                for (const link of node.links) {
                    visibleLinksSet.add(link);
                }
            }
        }
        
        renderLinks = Array.from(visibleLinksSet);
        // Sort subset by importance to maintain LOD logic
        renderLinks.sort((a, b) => b._importance - a._importance);
    }

    // --- SORT LINKS (OPTIMIZED) ---
    for (const link of renderLinks) {
        // Culling
        const sourceVisible = link.source.x >= minX && link.source.x <= maxX && link.source.y >= minY && link.source.y <= maxY;
        const targetVisible = link.target.x >= minX && link.target.x <= maxX && link.target.y >= minY && link.target.y <= maxY;

        // OPTIMIZATION: Strict culling as requested.
        // Only render edges if BOTH nodes are within the render area (screen + margin).
        if (!sourceVisible || !targetVisible) continue;

        const isHovered = highlightedNode && (link.source === highlightedNode || link.target === highlightedNode);
        const isActive = (activationLevels.get(String(link.source.id))||0) > 0.1 || (activationLevels.get(String(link.target.id))||0) > 0.1;

        if (isHovered || isActive) {
            // Limit hover context to avoid clutter on super-hubs
            if (highDetailZoom) {
                // With strict culling, we just need a safety cap
                if (activeLinks.length >= MAX_ACTIVE_EDGES) continue;
            } else {
                const limit = isHovered ? MAX_HOVER_EDGES : MAX_ACTIVE_EDGES;
                if (activeLinks.length >= limit) continue;
            }
            activeLinks.push({link, isActive});
            continue;
        }

        if (highlightedNode) continue; 
        if (ultraZoom) continue; 


        // OPTIMIZATION: Hard limit on visible background edges
        // Since links are sorted by importance, we get the best structure
        // When zoomed in, show all edges in viewport
        if (highDetailZoom) {
            if (totalRenderedEdges >= MAX_ZOOMED_EDGES) continue;
        } else {
            if (totalRenderedEdges >= MAX_VISIBLE_EDGES) continue;
        }

        totalRenderedEdges++;

        // OPTIMIZATION: Use pre-calculated flags
        if (link._isContra) {
            contraLinks.push(link);
            continue; 
        }

        if (hideLeaves && link._isLeafLink) continue;
        if (extremeZoom && !link._isHubLink) continue;

        if (link._isLeafLink) leafLinks.push(link);
        else hubLinks.push(link);
    }

    // Helper for distance fading
    const drawFadedBatch = (links, color, widthBase) => {
        ctx.strokeStyle = color;
        // Fix visibility on zoom out: ensure minimum screen pixels
        const minPixels = widthBase < 0.6 ? 1.0 : 2.0; // Leaves vs Hubs
        ctx.lineWidth = Math.max(widthBase, minPixels) / Math.max(transform.k, 0.01);
        
        const opaque = [];
        const faded = [];

        for (const l of links) {
            const dx = l.target.x - l.source.x;
            const dy = l.target.y - l.source.y;
            const dist = Math.sqrt(dx*dx + dy*dy);
            l._dist = dist; // Cache for drawing
            
            if (dist > 300) faded.push(l);
            else opaque.push(l);
        }

        // Batch draw opaque edges (Huge performance win)
        if (opaque.length > 0) {
            ctx.globalAlpha = 1.0;
            ctx.beginPath();
            for (const l of opaque) drawPath(l, false, l._dist);
            ctx.stroke();
        }

        // Draw faded edges individually
        if (faded.length > 0) {
            for (const l of faded) {
                const alpha = Math.max(0.3, 1.0 - (l._dist - 300) / 1000);
            ctx.globalAlpha = alpha;
            ctx.beginPath();
                drawPath(l, false, l._dist);
            ctx.stroke();
            }
        }
        ctx.globalAlpha = 1.0;
    };

    // --- BATCH 1: LEAF LINKS ---
    if (leafLinks.length > 0) {
        drawFadedBatch(leafLinks, CONFIG.color.edge.leaf, 0.5);
    }

    // --- BATCH 2: HUB LINKS ---
    if (hubLinks.length > 0) {
        drawFadedBatch(hubLinks, CONFIG.color.edge.hub, 0.8);
    }

    // --- BATCH 3: CONTRADICTION LINKS (RED) ---
    if (contraLinks.length > 0) {
        ctx.beginPath();
        for (const l of contraLinks) drawPath(l, false); 
        ctx.strokeStyle = '#f87171'; 
        ctx.lineWidth = 1.5 / Math.max(transform.k, 0.01); 
        ctx.setLineDash([5, 5]); 
        ctx.stroke();
        ctx.setLineDash([]); 
    }

    // --- BATCH 4: ACTIVE LINKS ---
    for (const al of activeLinks) {
        // OPTIMIZATION: Pre-calc dist for drawPath to avoid recalculation
        const dx = al.link.target.x - al.link.source.x;
        const dy = al.link.target.y - al.link.source.y;
        const dist = Math.sqrt(dx*dx + dy*dy);

        ctx.beginPath();
        drawPath(al.link, false, dist); 
        if (al.link.type === 'contradicts') {
            ctx.strokeStyle = '#ef4444'; 
        } else if (al.isActive) {
            ctx.strokeStyle = CONFIG.color.edge.active;
        } else {
            ctx.strokeStyle = CONFIG.color.edge.highlight;
        }
        
        // OPTIMIZATION: Thinner, more organic lines (neural connection style)
        const widthBase = al.isActive ? 1.0 + (pulse * 0.8) : 0.5 + (pulse * 0.5);
        ctx.lineWidth = Math.max(widthBase, 0.5) / Math.max(transform.k, 0.01);
        ctx.stroke();
    }

    let activeFont = ""; 

    // --- DRAW NODES ---
    for (const node of renderNodes) {
        // If using spatial index, renderNodes are already culled
        if (!useSpatialIndex) {
            if (node.x < minX || node.x > maxX || node.y < minY || node.y > maxY) continue;
        }

        const activeVal = activationLevels.get(String(node.id)) || 0;
        const isHovered = node === highlightedNode;
        const isNeighbor = highlightedNode && linkedNeighbors.has(node.id);
        const isSelected = node.id === selectedNode;
        
        // OPTIMIZATION: Use pre-calculated flags
        // "Render actual nodes only when we hover over" -> Hide leaves by default unless active/hovered
        // We treat "actual nodes" as the leaf content nodes. Hubs remain as structure.
        // When zoomed in, show leaf nodes
        if (!highDetailZoom && node._isLeaf && !activeVal && !isHovered && !isSelected && !isNeighbor) continue;
        
        if (extremeZoom && !node._isHub && !activeVal) continue;
        if (ultraZoom && node.degree <= 8 && !activeVal) continue; 

        if (highlightedNode && !isHovered && !isNeighbor && activeVal < 0.1) {
            ctx.globalAlpha = 0.1;
        } else {
            ctx.globalAlpha = 1.0;
        }

        let r = node._isLeaf ? 3 : (5 + Math.sqrt(node.degree));
        if (activeVal > 0) r += activeVal * (6 + pulse * 4); // Pulsing size for active nodes
        if (isHovered) r *= 1.3;

        if (extremeZoom) r = Math.max(r, 2 / transform.k);

        ctx.beginPath();
        ctx.arc(node.x, node.y, r, 0, 2 * Math.PI);
        
        // --- BIOLOGICAL GLOWS ---
        if (activeVal > 0.05) {
            const alpha = Math.min(activeVal, 0.8);
            
            // OPTIMIZATION: Use Gradient instead of ShadowBlur (Much faster)
            const glowRadius = r * (1.2 + pulse * 0.5);
            const grd = ctx.createRadialGradient(node.x, node.y, r * 0.5, node.x, node.y, glowRadius);
            grd.addColorStop(0, `rgba(${CONFIG.color.node.active}, ${alpha * 0.6})`);
            grd.addColorStop(1, `rgba(${CONFIG.color.node.active}, 0)`);
            
            ctx.fillStyle = grd;
            ctx.beginPath();
            ctx.arc(node.x, node.y, glowRadius, 0, 2 * Math.PI);
            ctx.fill();
            
            ctx.fillStyle = `rgba(${CONFIG.color.node.active}, ${alpha})`;
            ctx.beginPath(); // Reset path for core node
            ctx.arc(node.x, node.y, r, 0, 2 * Math.PI);
        } else {
             if (isHovered || isSelected) {
                ctx.fillStyle = "#fff";
                ctx.shadowBlur = 20;
                ctx.shadowColor = 'rgba(255, 255, 255, 0.5)';
            }
            // Use _isContra flag logic via raw data if strictly needed, 
            // but for nodes usually color by hub/leaf status
            else if (node._isLeaf) {
                ctx.fillStyle = CONFIG.color.node.leaf;
                ctx.shadowBlur = 0;
            } else {
                // Hub Node - Strong Neuron Glow (Simple Circle for Performance)
                // Gradients kill performance on zoom out with many nodes
                const glowRadius = r * 1.5;
                ctx.fillStyle = 'rgba(99, 102, 241, 0.3)';
                ctx.beginPath();
                ctx.arc(node.x, node.y, glowRadius, 0, 2 * Math.PI);
                ctx.fill();

                ctx.fillStyle = CONFIG.color.node.hub;
                ctx.beginPath(); // Reset for core
                ctx.arc(node.x, node.y, r, 0, 2 * Math.PI);
                
                ctx.shadowBlur = 0; 
            }
        }

        ctx.fill();
        ctx.shadowBlur = 0; // Reset for next element

        const showLabel = isHovered || isSelected || activeVal > 0.1 || (node._isHub && transform.k > 0.7);
        if (showLabel && !ultraZoom) { 
            let textAlpha = 0.2; 
            if (isHovered || isSelected || activeVal > 0.1) {
                textAlpha = 1.0;
            }

            ctx.fillStyle = `rgba(${CONFIG.color.node.text}, ${textAlpha})`;
            
            const targetFont = isHovered ? "600 12px Space Grotesk" : "500 10px Space Grotesk";
            if (activeFont !== targetFont) {
                ctx.font = targetFont;
                activeFont = targetFont;
            }
            ctx.fillText(node.label, node.x + r + 4, node.y + 3);
        }
    }

    ctx.restore();
}

function drawPath(link, simplify, preCalcDist) {
    const src = link.source;
    const tgt = link.target;
    if(isNaN(src.x) || isNaN(tgt.x)) return;

    ctx.moveTo(src.x, src.y);

    if (simplify) {
        ctx.lineTo(tgt.x, tgt.y);
        return;
    }

    // Asymmetric curve for more organic feel
    const mx = src.x * 0.4 + tgt.x * 0.6;
    const my = src.y * 0.4 + tgt.y * 0.6;
    const dx = tgt.x - src.x;
    const dy = tgt.y - src.y;
    const dist = preCalcDist !== undefined ? preCalcDist : Math.sqrt(dx*dx + dy*dy);
    
    if(dist === 0) return;

    const nx = -dy / dist;
    const ny = dx / dist;

    const cpX = mx + nx * link.curveOffset;
    const cpY = my + ny * link.curveOffset;

    ctx.quadraticCurveTo(cpX, cpY, tgt.x, tgt.y);
}

// ==========================================
// 5. INTERACTION & INSPECTOR
// ==========================================

function handleMouseMove(e) {
    const [simX, simY] = transform.invert(d3.pointer(e));
    const closest = simulation.find(simX, simY, 20 / transform.k); 

    if(closest !== highlightedNode) {
        highlightedNode = closest;
        linkedNeighbors.clear();
        
        if(closest) {
            links.forEach(l => {
                if(l.source === closest) linkedNeighbors.add(l.target.id);
                if(l.target === closest) linkedNeighbors.add(l.source.id);
            });
            document.body.style.cursor = 'pointer';
        } else {
            document.body.style.cursor = 'default';
        }
    }
}

function handleClick(e) {
    if(highlightedNode) openInspector(highlightedNode);
    else closeInspector();
}

function dragSubject(e) {
    const [simX, simY] = transform.invert(d3.pointer(e));
    return simulation.find(simX, simY, 30 / transform.k);
}

function dragStarted(e) {
    if (!e.active) simulation.alphaTarget(0.3).restart();
    e.subject.fx = e.subject.x;
    e.subject.fy = e.subject.y;
}

function dragged(e) {
    const [simX, simY] = transform.invert(d3.pointer(e));
    e.subject.fx = simX;
    e.subject.fy = simY;
}

function dragEnded(e) {
    if (!e.active) simulation.alphaTarget(0);
    e.subject.fx = null;
    e.subject.fy = null;
}

// === INSPECTOR LOGIC ===

function openInspector(node) {
    selectedNode = node.id;
    const raw = node.raw || {};
    document.getElementById('node-inspector').classList.add('open');
    document.getElementById('ins-title').innerText = "Belief Node";

    const setText = (id, val) => {
        const el = document.getElementById(id);
        if(el) el.innerText = val !== undefined && val !== null ? val : '-';
    };

    setText('meta-id', node.id);
    setText('meta-type', raw.type || 'Concept');
    
    setText('meta-sub', raw.subject);
    setText('meta-pred', raw.predicate);
    setText('meta-obj', raw.object);
    
    setText('score-evidence', raw.evidence_score?.toFixed(2));
    setText('score-contra', raw.contradiction_score?.toFixed(2));
    setText('score-struct', raw.structural_support_score?.toFixed(2));
    setText('score-usage', raw.usage_count);
    
    setText('meta-decay', raw.decay_rate);

    const list = document.getElementById('conn-list');
    list.innerHTML = '';
    
    const connected = links.filter(l => l.source.id === node.id || l.target.id === node.id);
    
    const connHeader = document.createElement('div');
    connHeader.style.cssText = "font-size:0.6rem; color:#94a3b8; letter-spacing:0.1em; margin:10px 0 4px 0;";
    connHeader.innerText = `${connected.length} LINKED NODES`;
    list.appendChild(connHeader);

    connected.slice(0, 10).forEach(l => {
        const other = l.source.id === node.id ? l.target : l.source;
        const row = document.createElement('div');
        row.className = 'conn-row';
        row.innerHTML = `<div>${other.label}</div>`;
        row.onclick = () => {
            openInspector(other);
            focusOnNode(other); 
        };
        list.appendChild(row);
    });
}

function closeInspector() {
    document.getElementById('node-inspector').classList.remove('open');
    selectedNode = null;
}

function setupUI() {
    document.getElementById('send-btn').onclick = sendMessage;
    document.getElementById('user-input').onkeydown = (e) => { 
        if(e.key==='Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); } 
    };

    document.querySelectorAll('.toggle-btn').forEach(btn => {
        btn.onclick = () => {
            btn.classList.toggle('active');
            if(btn.id === 'btn-auto') socket.emit('toggle_autonomous', { enabled: btn.classList.contains('active') });
        };
    });
    
    // Rendering Toggle Logic
    const btnVisualizer = document.getElementById('btn-visualizer');
    let lastSidebarWidth = '';
    
    if (btnVisualizer) {
        btnVisualizer.onclick = () => {
            renderingEnabled = !renderingEnabled;
            btnVisualizer.classList.toggle('active', renderingEnabled);
            
            if (!renderingEnabled) {
                lastSidebarWidth = document.getElementById('sidebar').style.width;
                document.body.classList.add('no-render');
                if (simulation) simulation.stop();
            } else {
                const loader = document.getElementById('visualizer-loader');
                if (loader) loader.style.display = 'flex';

                // Use setTimeout to allow the loader to render before the heavy lift
                setTimeout(() => {
                    document.body.classList.remove('no-render');
                    if(lastSidebarWidth) document.getElementById('sidebar').style.width = lastSidebarWidth;
                    resize(); 
                    if (simulation) simulation.alpha(0.1).restart();
                    
                    // Hide loader after a brief moment to ensure smooth transition
                    setTimeout(() => { if (loader) loader.style.display = 'none'; }, 400);
                }, 50);
            }
        };
    }

    // Sidebar Drag/Resize Logic
    const handle = document.getElementById('drag-handle');
    const sidebar = document.getElementById('sidebar');
    let isResizing = false;

    if (handle && sidebar) {
        handle.addEventListener('mousedown', (e) => {
            isResizing = true;
            document.body.style.cursor = 'col-resize';
            e.preventDefault();
        });
        document.addEventListener('mousemove', (e) => {
            if (!isResizing || !renderingEnabled) return;
            let newWidth = e.clientX;
            if (newWidth < 300) newWidth = 300;
            if (newWidth > window.innerWidth - 100) newWidth = window.innerWidth - 100;
            sidebar.style.width = `${newWidth}px`;
            requestAnimationFrame(resize);
        });
        document.addEventListener('mouseup', () => {
            if (isResizing) {
                isResizing = false;
                document.body.style.cursor = 'default';
                resize();
            }
        });
    }

    document.querySelector('.ins-close').onclick = closeInspector;

    const clusterToggle = document.getElementById('cluster-toggle');
    const clusterNav = document.getElementById('cluster-nav');
    const clusterHeader = document.getElementById('cluster-header');
    
    if(clusterHeader) {
        clusterHeader.onclick = () => {
            if(clusterNav.style.maxHeight === '45px') {
                clusterNav.style.maxHeight = '300px';
                clusterToggle.innerText = '▼';
            } else {
                clusterNav.style.maxHeight = '45px';
                clusterToggle.innerText = '▲';
            }
        };
    }

    document.getElementById('phy-toggle').onclick = function() {
        this.classList.toggle('active');
        document.getElementById('physics-controls').classList.toggle('visible');
    };
    
    const rngRep = document.getElementById('rng-rep');
    const rngDist = document.getElementById('rng-dist');
    
    if(rngRep) rngRep.oninput = (e) => { 
        CONFIG.physics.repulsion = -Math.abs(+e.target.value); 
        initSimulation(); simulation.alpha(0.3).restart(); 
    };
    if(rngDist) rngDist.oninput = (e) => { 
        CONFIG.physics.distance = +e.target.value; 
        initSimulation(); simulation.alpha(0.3).restart(); 
    };
}

function updateStatus(text, active=false) {
    const pill = document.getElementById('status-pill');
    pill.innerHTML = `<span class="brand-dot" style="width:6px;height:6px;"></span> ${text}`;
    if(active) pill.classList.add('active'); else pill.classList.remove('active');
}

function addMessage(text, type) {
    const div = document.createElement('div');
    div.className = `msg ${type}`;
    div.innerHTML = text;
    const hist = document.getElementById('chat-history');
    hist.appendChild(div);
    hist.scrollTop = hist.scrollHeight;
}

function sendMessage() {
    const input = document.getElementById('user-input');
    const txt = input.value.trim();
    if(!txt) return;
    addMessage(txt, 'user');
    input.value = '';
    
    const v = document.getElementById('btn-verbose')?.classList.contains('active');
    const d = document.getElementById('btn-deep')?.classList.contains('active');
    
    socket.emit('chat_message', { message: txt, verbose: v, deep_think: d });
    updateStatus("Processing...", true);
}

socket.on('status_update', (d) => updateStatus(d.status, true));

socket.on('system_message', (data) => {
    addMessage(`🔎 ${data.message}`, 'ai');
});

socket.on('beliefs_activated', (ids) => {
    if (!Array.isArray(ids)) return;
    ids.sort((a, b) => (b.score || 0) - (a.score || 0));
    ids.forEach((nodeInfo, index) => {
        setTimeout(() => {
            activationLevels.set(String(nodeInfo.id), 1.0);
        }, index * 50);
    });
});

socket.on('belief_created', (newNode) => {
    const nodeExists = nodes.some(n => n.id === newNode.id);
    if (nodeExists) return;
    
    const displayName = newNode.object ? newNode.object : newNode.subject;

    nodes.push({
        id: newNode.id,
        label: displayName,
        raw: newNode,
        x: selectedNode ? nodes.find(n=>n.id===selectedNode)?.x + (Math.random()-0.5)*50 : (Math.random() - 0.5) * 500,
        y: selectedNode ? nodes.find(n=>n.id===selectedNode)?.y + (Math.random()-0.5)*50 : (Math.random() - 0.5) * 500,
        degree: 0
    });

    activationLevels.set(String(newNode.id), 1.0);
    updateClusterNav(); 

    if (simulation && renderingEnabled) {
        simulation.nodes(nodes);
        simulation.alpha(0.3).restart();
    }
});

socket.on('edge_created', (newEdge) => {
    const sourceNode = nodes.find(n => n.id === String(newEdge.source_id));
    const targetNode = nodes.find(n => n.id === String(newEdge.target_id));

    if (sourceNode && targetNode) {
        const baseOffset = ((parseInt(newEdge.source_id) + parseInt(newEdge.target_id)) % 100) - 50;
        links.push({
            source: sourceNode,
            target: targetNode,
            curveOffset: baseOffset * 2.5
        });
        sourceNode.degree++;
        targetNode.degree++;
        
        updateClusterNav(); 

        if (simulation && renderingEnabled) {
            simulation.force("link").links(links);
            simulation.alpha(0.3).restart();
        }
    }
});

socket.on('belief_strengthened', (data) => {
    activationLevels.set(String(data.id), 1.0);
});

socket.on('toggle_autonomous', (d) => {
    document.getElementById('auto-badge').classList.toggle('visible', d.enabled);
});

socket.on('graph_update', (data) => {
    if(data && data.nodes) mergeGraphData(data);
    else loadGraphData();
});

socket.on('token_update', (data) => {
    const el = document.getElementById('token-count');
    const container = document.getElementById('token-pill');
    
    // Update number
    totalTokens += (data.total || 0);
    el.innerText = totalTokens.toLocaleString();
    
    // Visual flash effect on update
    container.style.borderColor = 'rgba(99, 102, 241, 0.6)';
    container.style.color = '#e2e8f0';
    
    setTimeout(() => {
        container.style.borderColor = 'rgba(99, 102, 241, 0.1)';
        container.style.color = '#64748b';
    }, 500);
});

socket.on('chat_response', (d) => { 
    addMessage(d.response, 'ai', d.reasoning, d.logs); 
    updateStatus("Idle"); 
});

function addMessage(text, type, reasoning = null, logs = null) {
    const hist = document.getElementById('chat-history');
    const div = document.createElement('div');
    div.className = `msg ${type}`;
    
    let content = `<div class="msg-text">${text}</div>`;

    // 1. Reasoning Section (Deep Think)
    if (reasoning) {
        content += `
            <details class="msg-details deep-think">
                <summary>🧠 Reasoning Process</summary>
                <div class="details-content">${reasoning.replace(/\n/g, '<br>')}</div>
            </details>
        `;
    }

    // 2. Logs Section (Verbose)
    if (logs && logs.length > 0) {
        content += `
            <details class="msg-details verbose-logs">
                <summary>📜 System Logs</summary>
                <div class="details-content code-block">${logs.join('<br>')}</div>
            </details>
        `;
    }

    div.innerHTML = content;
    hist.appendChild(div);
    hist.scrollTop = hist.scrollHeight;
}