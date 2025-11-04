// Initialize the 3D scene
let scene, camera, renderer, controls;
let graph = { nodes: [], links: [] };

// Initialize the dashboard
document.addEventListener('DOMContentLoaded', () => {
    initializeCodeMap();
    initializeDashboard();
    
    // Fetch initial data
    fetchDashboardData();
    fetchCodeMapData();
    
    // Set up periodic updates
    setInterval(fetchDashboardData, 300000); // Update every 5 minutes
});

// Dashboard Initialization and Data Handling
async function initializeDashboard() {
    const dashboardContent = document.getElementById('dashboard-content');
    
    // Create sections for different metrics
    dashboardContent.innerHTML = `
        <div id="health-metrics"></div>
        <div id="alerts-section"></div>
        <div id="predictions-section"></div>
    `;
}

async function fetchDashboardData() {
    try {
        const response = await fetch('/api/insights/dashboard');
        const data = await response.json();
        
        updateHealthMetrics(data.metrics);
        updateAlerts(data.alerts);
        updatePredictions(data.predictions);
        updateSubsystemHealth(data.subsystem_health);
    } catch (error) {
        console.error('Error fetching dashboard data:', error);
    }
}

function updateHealthMetrics(metrics) {
    const healthMetrics = document.getElementById('health-metrics');
    
    healthMetrics.innerHTML = `
        <div class="metric-card">
            <div class="metric-title">Overall Health Score</div>
            <div class="metric-value">${Math.round(metrics.maintainability_index)}</div>
            ${getHealthIndicator(metrics.maintainability_index)}
        </div>
        <div class="metric-card">
            <div class="metric-title">Test Coverage</div>
            <div class="metric-value">${Math.round(metrics.test_coverage)}%</div>
            ${getHealthIndicator(metrics.test_coverage)}
        </div>
        <div class="metric-card">
            <div class="metric-title">Complexity Score</div>
            <div class="metric-value">${metrics.complexity_score.toFixed(1)}</div>
            ${getHealthIndicator(100 - metrics.complexity_score * 10)}
        </div>
    `;
}

function updateAlerts(alerts) {
    const alertsSection = document.getElementById('alerts-section');
    
    alertsSection.innerHTML = `
        <h3>Active Alerts</h3>
        ${alerts.map(alert => `
            <div class="alert-card ${alert.severity}">
                <div class="alert-header">
                    <span class="alert-severity ${alert.severity}">${alert.severity}</span>
                </div>
                <div class="alert-message">${alert.message}</div>
                <div class="alert-details">
                    <p><strong>Impact:</strong> ${alert.impact}</p>
                    <p><strong>Action:</strong> ${alert.suggested_action}</p>
                </div>
            </div>
        `).join('')}
    `;
}

function updatePredictions(predictions) {
    const predictionsSection = document.getElementById('predictions-section');
    
    predictionsSection.innerHTML = `
        <h3>Predictions & Insights</h3>
        <div class="metric-card">
            <div class="metric-title">Complexity Trend</div>
            <div class="metric-value">${predictions.complexity_trend.trend}</div>
            <div class="metric-trend">
                Rate: ${predictions.complexity_trend.rate.toFixed(2)} per week
            </div>
        </div>
        <div class="metric-card">
            <div class="metric-title">Maintenance Needs</div>
            <div class="metric-value">${predictions.maintenance_needs.priority}</div>
            <div class="metric-trend">
                Timeline: ${predictions.maintenance_needs.recommended_timeline}
            </div>
        </div>
    `;
}

function updateSubsystemHealth(subsystems) {
    const healthSection = document.getElementById('health-metrics');
    
    const subsystemCards = subsystems.map(subsystem => `
        <div class="metric-card">
            <div class="metric-title">${subsystem.name}</div>
            <div class="metric-value">${Math.round(subsystem.score)}</div>
            <div class="metric-trend">
                ${getHealthIndicator(subsystem.score)}
                ${subsystem.issues.length > 0 ? `
                    <span class="issues-count">${subsystem.issues.length} issues</span>
                ` : ''}
            </div>
        </div>
    `).join('');
    
    healthSection.insertAdjacentHTML('beforeend', subsystemCards);
}

function getHealthIndicator(score) {
    let className = 'health-critical';
    if (score >= 80) {
        className = 'health-good';
    } else if (score >= 60) {
        className = 'health-warning';
    }
    
    return `<span class="health-indicator ${className}"></span>`;
}

// 3D Code Map Initialization and Handling
async function initializeCodeMap() {
    // Set up Three.js scene
    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(75, window.innerWidth / 2 / window.innerHeight, 0.1, 1000);
    
    renderer = new THREE.WebGLRenderer({ antialias: true });
    const container = document.getElementById('codemap');
    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);
    
    // Add lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);
    
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
    directionalLight.position.set(10, 10, 10);
    scene.add(directionalLight);
    
    // Set up camera position
    camera.position.z = 50;
    
    // Add OrbitControls
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    
    // Start animation loop
    animate();
    
    // Handle window resize
    window.addEventListener('resize', onWindowResize, false);
}

async function fetchCodeMapData() {
    try {
        const response = await fetch('/api/code_map/generate');
        const data = await response.json();
        
        // Update graph data
        graph = {
            nodes: data.nodes,
            links: data.links
        };
        
        updateCodeMap();
    } catch (error) {
        console.error('Error fetching code map data:', error);
    }
}

function updateCodeMap() {
    // Clear existing objects
    while(scene.children.length > 0) { 
        scene.remove(scene.children[0]); 
    }
    
    // Create nodes
    graph.nodes.forEach(node => {
        const geometry = new THREE.SphereGeometry(node.size || 1);
        const material = new THREE.MeshPhongMaterial({
            color: getNodeColor(node.health_score)
        });
        
        const sphere = new THREE.Mesh(geometry, material);
        sphere.position.set(node.x, node.y, node.z);
        sphere.userData = node;
        
        scene.add(sphere);
    });
    
    // Create links
    graph.links.forEach(link => {
        const start = graph.nodes[link.source];
        const end = graph.nodes[link.target];
        
        const points = [];
        points.push(new THREE.Vector3(start.x, start.y, start.z));
        points.push(new THREE.Vector3(end.x, end.y, end.z));
        
        const geometry = new THREE.BufferGeometry().setFromPoints(points);
        const material = new THREE.LineBasicMaterial({
            color: 0x999999,
            opacity: 0.5,
            transparent: true
        });
        
        const line = new THREE.Line(geometry, material);
        scene.add(line);
    });
}

function getNodeColor(healthScore) {
    if (healthScore >= 80) {
        return 0x27ae60; // Green
    } else if (healthScore >= 60) {
        return 0xf1c40f; // Yellow
    } else {
        return 0xe74c3c; // Red
    }
}

function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}

function onWindowResize() {
    const container = document.getElementById('codemap');
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
}

// Tooltip handling
let tooltip = null;

function showTooltip(event, data) {
    if (!tooltip) {
        tooltip = document.createElement('div');
        tooltip.className = 'tooltip';
        document.body.appendChild(tooltip);
    }
    
    tooltip.innerHTML = `
        <strong>${data.name}</strong><br>
        Health Score: ${Math.round(data.health_score)}<br>
        ${data.issues ? `Issues: ${data.issues.length}` : ''}
    `;
    
    tooltip.style.left = event.pageX + 10 + 'px';
    tooltip.style.top = event.pageY + 10 + 'px';
    tooltip.style.display = 'block';
}

function hideTooltip() {
    if (tooltip) {
        tooltip.style.display = 'none';
    }
}