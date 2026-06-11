/* ╔══════════════════════════════════════════════════════════════╗
   ║  NetFlow AI — Dashboard Application Logic                   ║
   ║  Industrial Network Traffic Predictive Forecaster           ║
   ╚══════════════════════════════════════════════════════════════╝ */

// ────────────────────────────────────────────────────────────────
// CONSTANTS
// ────────────────────────────────────────────────────────────────
const API_BASE = 'http://localhost:8000';

const CHART_COLORS = {
    cyan:       'rgba(0, 212, 255, 1)',
    cyanFaded:  'rgba(0, 212, 255, 0.08)',
    violet:     'rgba(124, 58, 237, 1)',
    violetFaded:'rgba(124, 58, 237, 0.08)',
    amber:      'rgba(245, 158, 11, 1)',
    amberFaded: 'rgba(245, 158, 11, 0.08)',
    red:        'rgba(239, 68, 68, 0.25)',
    grid:       'rgba(148, 163, 184, 0.06)',
    gridBorder: 'rgba(148, 163, 184, 0.08)',
    tooltip:    'rgba(2, 6, 18, 0.95)',
    text:       'rgba(148, 163, 184, 1)',
    textFaint:  'rgba(100, 116, 139, 1)',
};

const CONGESTION_THRESHOLDS = {
    low:    0.3,
    medium: 0.6,
    high:   0.8,
};

// ────────────────────────────────────────────────────────────────
// APPLICATION STATE
// ────────────────────────────────────────────────────────────────
let historicalData = [];
let predictionData = [];
let historicalChart = null;
let forecastChart = null;
let isLoading = false;


// ────────────────────────────────────────────────────────────────
// DOM REFERENCES
// ────────────────────────────────────────────────────────────────
const dom = {
    throughputValue:    () => document.getElementById('throughput-value'),
    throughputTrend:    () => document.getElementById('throughput-trend'),
    peakValue:          () => document.getElementById('peak-value'),
    peakTrend:          () => document.getElementById('peak-trend'),
    congestionValue:    () => document.getElementById('congestion-value'),
    congestionDot:      () => document.getElementById('congestion-dot'),
    congestionLevel:    () => document.getElementById('congestion-level'),
    actionValue:        () => document.getElementById('action-value'),
    actionDetail:       () => document.getElementById('action-detail'),
    groqApiKey:         () => document.getElementById('groq-api-key'),
    toggleApiKey:       () => document.getElementById('toggle-api-key'),
    eyeShow:            () => document.getElementById('eye-icon-show'),
    eyeHide:            () => document.getElementById('eye-icon-hide'),
    horizonSlider:      () => document.getElementById('prediction-horizon'),
    horizonLabel:       () => document.getElementById('horizon-label'),
    btnGenerate:        () => document.getElementById('btn-generate'),
    btnSample:          () => document.getElementById('btn-sample'),
    historicalCanvas:   () => document.getElementById('historicalChart'),
    forecastCanvas:     () => document.getElementById('forecastChart'),
    severityMarker:     () => document.getElementById('severity-marker'),
    insightsPlaceholder:() => document.getElementById('insights-placeholder'),
    insightsLoading:    () => document.getElementById('insights-loading'),
    insightsResult:     () => document.getElementById('insights-result'),
    analysisText:       () => document.getElementById('analysis-text'),
    recommendationsList:() => document.getElementById('recommendations-list'),
    toastContainer:     () => document.getElementById('toast-container'),
};


// ────────────────────────────────────────────────────────────────
// INITIALIZATION
// ────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    initCharts();
    bindEvents();
    loadSampleData();
});


function bindEvents() {
    // API key visibility toggle
    dom.toggleApiKey().addEventListener('click', () => {
        const input = dom.groqApiKey();
        const isPassword = input.type === 'password';
        input.type = isPassword ? 'text' : 'password';
        dom.eyeShow().classList.toggle('hidden', isPassword);
        dom.eyeHide().classList.toggle('hidden', !isPassword);
    });

    // Horizon slider
    dom.horizonSlider().addEventListener('input', (e) => {
        const steps = parseInt(e.target.value, 10);
        const minutes = steps * 5;
        let label;
        if (minutes < 60) {
            label = `${steps} steps (${minutes} min)`;
        } else {
            const hours = (minutes / 60);
            label = `${steps} steps (${hours % 1 === 0 ? hours : hours.toFixed(1)} hour${hours > 1 ? 's' : ''})`;
        }
        dom.horizonLabel().textContent = label;
    });

    // Generate button
    dom.btnGenerate().addEventListener('click', generateForecast);

    // Sample data button
    dom.btnSample().addEventListener('click', loadSampleData);

    // Nav link active state
    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', (e) => {
            document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
            e.currentTarget.classList.add('active');
        });
    });
}


// ────────────────────────────────────────────────────────────────
// CHART INITIALIZATION
// ────────────────────────────────────────────────────────────────
function createGradient(ctx, colorStart, colorEnd) {
    const gradient = ctx.createLinearGradient(0, 0, 0, ctx.canvas.clientHeight);
    gradient.addColorStop(0, colorStart);
    gradient.addColorStop(1, colorEnd);
    return gradient;
}

function getChartDefaults() {
    return {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
            mode: 'index',
            intersect: false,
        },
        plugins: {
            legend: {
                display: true,
                position: 'top',
                align: 'end',
                labels: {
                    color: CHART_COLORS.text,
                    font: {
                        family: "'Inter', sans-serif",
                        size: 11,
                        weight: '500',
                    },
                    boxWidth: 12,
                    boxHeight: 3,
                    borderRadius: 2,
                    useBorderRadius: true,
                    padding: 16,
                },
            },
            tooltip: {
                backgroundColor: CHART_COLORS.tooltip,
                titleColor: '#f1f5f9',
                bodyColor: '#94a3b8',
                borderColor: 'rgba(255,255,255,0.08)',
                borderWidth: 1,
                cornerRadius: 8,
                padding: 12,
                titleFont: {
                    family: "'Inter', sans-serif",
                    size: 12,
                    weight: '600',
                },
                bodyFont: {
                    family: "'Inter', sans-serif",
                    size: 11,
                },
                displayColors: true,
                boxWidth: 8,
                boxHeight: 8,
                boxPadding: 4,
                usePointStyle: true,
            },
        },
        scales: {
            x: {
                grid: {
                    color: CHART_COLORS.grid,
                    drawBorder: false,
                },
                ticks: {
                    color: CHART_COLORS.textFaint,
                    font: {
                        family: "'Inter', sans-serif",
                        size: 10,
                    },
                    maxRotation: 0,
                    autoSkipPadding: 20,
                },
            },
            y: {
                grid: {
                    color: CHART_COLORS.grid,
                    drawBorder: false,
                },
                ticks: {
                    color: CHART_COLORS.textFaint,
                    font: {
                        family: "'Inter', sans-serif",
                        size: 10,
                    },
                    padding: 8,
                },
                beginAtZero: true,
            },
        },
        elements: {
            line: {
                tension: 0.4,
                borderWidth: 2,
            },
            point: {
                radius: 0,
                hoverRadius: 4,
                hoverBorderWidth: 2,
                hoverBackgroundColor: '#fff',
            },
        },
    };
}

function initCharts() {
    // ── Historical Chart ──
    const histCtx = dom.historicalCanvas().getContext('2d');
    const histGradCyan = createGradient(histCtx, CHART_COLORS.cyanFaded, 'rgba(0,212,255,0)');
    const histGradViolet = createGradient(histCtx, CHART_COLORS.violetFaded, 'rgba(124,58,237,0)');
    const histGradAmber = createGradient(histCtx, CHART_COLORS.amberFaded, 'rgba(245,158,11,0)');

    historicalChart = new Chart(histCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Bandwidth (Mbps)',
                    data: [],
                    borderColor: CHART_COLORS.cyan,
                    backgroundColor: histGradCyan,
                    fill: true,
                    order: 1,
                },
                {
                    label: 'Packets (K/s)',
                    data: [],
                    borderColor: CHART_COLORS.violet,
                    backgroundColor: histGradViolet,
                    fill: true,
                    order: 2,
                },
                {
                    label: 'Latency (ms)',
                    data: [],
                    borderColor: CHART_COLORS.amber,
                    backgroundColor: histGradAmber,
                    fill: true,
                    order: 3,
                },
            ],
        },
        options: {
            ...getChartDefaults(),
            plugins: {
                ...getChartDefaults().plugins,
                title: {
                    display: false,
                },
            },
        },
    });

    // ── Forecast Chart ──
    const fcCtx = dom.forecastCanvas().getContext('2d');
    const fcGradCyan = createGradient(fcCtx, CHART_COLORS.cyanFaded, 'rgba(0,212,255,0)');

    forecastChart = new Chart(fcCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Actual Bandwidth',
                    data: [],
                    borderColor: CHART_COLORS.cyan,
                    backgroundColor: fcGradCyan,
                    fill: true,
                    order: 2,
                },
                {
                    label: 'Predicted Bandwidth',
                    data: [],
                    borderColor: CHART_COLORS.violet,
                    backgroundColor: 'transparent',
                    borderDash: [6, 4],
                    fill: false,
                    order: 1,
                },
                {
                    label: 'Congestion Zone',
                    data: [],
                    borderColor: 'rgba(239, 68, 68, 0.3)',
                    backgroundColor: CHART_COLORS.red,
                    fill: true,
                    borderWidth: 1,
                    borderDash: [4, 4],
                    pointRadius: 0,
                    pointHoverRadius: 0,
                    order: 3,
                },
            ],
        },
        options: {
            ...getChartDefaults(),
            plugins: {
                ...getChartDefaults().plugins,
                title: {
                    display: false,
                },
            },
        },
    });
}


// ────────────────────────────────────────────────────────────────
// API FUNCTIONS
// ────────────────────────────────────────────────────────────────
async function loadSampleData() {
    try {
        showToast('Loading sample data…', 'info');

        const response = await fetch(`${API_BASE}/api/sample-data`, {
            method: 'GET',
            headers: { 'Content-Type': 'application/json' },
            signal: AbortSignal.timeout(15000),
        });

        if (!response.ok) {
            throw new Error(`Server returned ${response.status}: ${response.statusText}`);
        }

        const result = await response.json();
        historicalData = result.data || result;

        // Update Historical Chart
        const labels = historicalData.map((_, i) => `T-${historicalData.length - i}`);
        const bandwidth = historicalData.map(d => d.bandwidth || d.Bandwidth || d[0] || 0);
        const packets = historicalData.map(d => d.packets || d.Packets || d[1] || 0);
        const latency = historicalData.map(d => d.latency || d.Latency || d[2] || 0);

        historicalChart.data.labels = labels;
        historicalChart.data.datasets[0].data = bandwidth;
        historicalChart.data.datasets[1].data = packets;
        historicalChart.data.datasets[2].data = latency;
        historicalChart.update('active');

        // Update metric cards with latest values
        const lastEntry = historicalData[historicalData.length - 1];
        const prevEntry = historicalData.length > 1
            ? historicalData[historicalData.length - 2]
            : lastEntry;

        const currentBandwidth = lastEntry.bandwidth || lastEntry.Bandwidth || lastEntry[0] || 0;
        const prevBandwidth = prevEntry.bandwidth || prevEntry.Bandwidth || prevEntry[0] || 0;
        const changePct = prevBandwidth > 0
            ? (((currentBandwidth - prevBandwidth) / prevBandwidth) * 100)
            : 0;

        animateValue(dom.throughputValue(), 0, currentBandwidth, 800);
        updateTrend(dom.throughputTrend(), changePct);

        showToast('Sample data loaded successfully!', 'success');

    } catch (error) {
        console.error('Failed to load sample data:', error);

        if (error.name === 'TimeoutError') {
            showToast('Request timed out. Is the backend running?', 'error');
        } else if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError')) {
            showToast('Cannot reach backend at ' + API_BASE + '. Using demo data.', 'error');
            loadDemoData();
        } else {
            showToast('Error loading data: ' + error.message, 'error');
            loadDemoData();
        }
    }
}


async function generateForecast() {
    if (isLoading) return;

    // Validate
    if (historicalData.length === 0) {
        showToast('Please load data first before generating predictions.', 'error');
        return;
    }

    const groqKey = dom.groqApiKey().value.trim();
    const horizon = parseInt(dom.horizonSlider().value, 10);

    setLoading(true);

    try {
        const payload = {
            data: historicalData,
            horizon: horizon,
        };

        if (groqKey) {
            payload.groq_api_key = groqKey;
        }

        const response = await fetch(`${API_BASE}/api/forecast`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
            signal: AbortSignal.timeout(60000),
        });

        if (!response.ok) {
            const errBody = await response.json().catch(() => ({}));
            throw new Error(errBody.detail || `Server error: ${response.status}`);
        }

        const result = await response.json();
        predictionData = result.predictions || result.forecast || [];

        // Update forecast chart
        updateForecastChart(predictionData, result.congestion_threshold);

        // Update metric cards
        const predictions = predictionData.map(d => d.bandwidth || d.Bandwidth || d[0] || d || 0);
        const peak = Math.max(...predictions);
        animateValue(dom.peakValue(), 0, peak, 800);

        const congestionProb = result.congestion_probability || result.congestion_risk || 0;
        animateValue(dom.congestionValue(), 0, Math.round(congestionProb * 100), 600);
        updateCongestionIndicator(congestionProb);

        const action = result.recommended_action || getRecommendedAction(congestionProb);
        dom.actionValue().textContent = action;
        dom.actionDetail().querySelector('.trend-text').textContent = 'Updated just now';

        // Display insights
        if (result.insights || result.analysis) {
            displayInsights(result.insights || result.analysis);
        }

        showToast('Forecast generated successfully!', 'success');

    } catch (error) {
        console.error('Forecast error:', error);

        if (error.name === 'TimeoutError') {
            showToast('Prediction request timed out. The model may need more time.', 'error');
        } else if (error.message.includes('Failed to fetch')) {
            showToast('Cannot connect to backend. Please check the server.', 'error');
        } else {
            showToast('Forecast error: ' + error.message, 'error');
        }
    } finally {
        setLoading(false);
    }
}


// ────────────────────────────────────────────────────────────────
// DEMO / FALLBACK DATA
// ────────────────────────────────────────────────────────────────
function loadDemoData() {
    const numPoints = 100;
    historicalData = [];

    for (let i = 0; i < numPoints; i++) {
        const t = i / numPoints;
        // Simulate realistic network traffic patterns
        const baseTraffic = 450 + 150 * Math.sin(2 * Math.PI * t * 3);
        const noise = (Math.random() - 0.5) * 80;
        const spike = (i > 60 && i < 75) ? 200 * Math.sin((i - 60) / 15 * Math.PI) : 0;

        historicalData.push({
            bandwidth: Math.max(100, baseTraffic + noise + spike),
            packets: Math.max(10, 85 + 30 * Math.sin(2 * Math.PI * t * 2.5) + (Math.random() - 0.5) * 20),
            latency: Math.max(1, 12 + 5 * Math.sin(2 * Math.PI * t * 4) + (Math.random() - 0.5) * 6),
        });
    }

    // Update Historical Chart
    const labels = historicalData.map((_, i) => `T-${historicalData.length - i}`);
    const bandwidth = historicalData.map(d => d.bandwidth);
    const packets = historicalData.map(d => d.packets);
    const latency = historicalData.map(d => d.latency);

    historicalChart.data.labels = labels;
    historicalChart.data.datasets[0].data = bandwidth;
    historicalChart.data.datasets[1].data = packets;
    historicalChart.data.datasets[2].data = latency;
    historicalChart.update('active');

    // Update metric cards
    const last = historicalData[historicalData.length - 1];
    const prev = historicalData[historicalData.length - 2];
    const changePct = ((last.bandwidth - prev.bandwidth) / prev.bandwidth) * 100;

    animateValue(dom.throughputValue(), 0, last.bandwidth, 800);
    updateTrend(dom.throughputTrend(), changePct);

    // Generate demo predictions
    generateDemoPredictions();
}


function generateDemoPredictions() {
    const horizon = parseInt(dom.horizonSlider().value, 10);
    const lastBw = historicalData[historicalData.length - 1].bandwidth;
    const predictions = [];

    for (let i = 0; i < horizon; i++) {
        const t = i / horizon;
        const trend = lastBw + (80 * Math.sin(2 * Math.PI * t * 2));
        const noise = (Math.random() - 0.5) * 40;
        predictions.push({
            bandwidth: Math.max(100, trend + noise),
        });
    }

    predictionData = predictions;
    const congestionThreshold = 650;

    updateForecastChart(predictions, congestionThreshold);

    const peak = Math.max(...predictions.map(p => p.bandwidth));
    animateValue(dom.peakValue(), 0, peak, 800);

    const congestionProb = predictions.filter(p => p.bandwidth > congestionThreshold).length / predictions.length;
    animateValue(dom.congestionValue(), 0, Math.round(congestionProb * 100), 600);
    updateCongestionIndicator(congestionProb);

    const action = getRecommendedAction(congestionProb);
    dom.actionValue().textContent = action;
    dom.actionDetail().querySelector('.trend-text').textContent = 'Based on demo data';

    // Show demo insights
    displayInsights({
        analysis: `**Traffic Pattern Analysis**\n\nThe network traffic exhibits cyclical patterns with a period of approximately 30 minutes, typical of batch processing workloads in industrial environments.\n\nPeak bandwidth observed at **${formatNumber(peak)} Mbps** with ${predictions.filter(p => p.bandwidth > congestionThreshold).length} predicted congestion events in the forecast window.\n\n**Key Observations:**\n• Traffic follows a sinusoidal pattern suggesting automated data transfers\n• Latency correlation with bandwidth spikes indicates potential buffer saturation\n• Packet rate remains relatively stable, suggesting large-payload transfers`,
        recommendations: [
            'Implement QoS policies to prioritize critical SCADA traffic during peak hours',
            'Consider traffic shaping to smooth burst patterns and reduce congestion risk',
            'Deploy packet-level monitoring on the primary gateway to identify large flows',
            'Evaluate link capacity upgrade if sustained throughput exceeds 600 Mbps',
            'Configure automated alerting for congestion probability exceeding 70%',
        ],
        severity: congestionProb,
    });
}


// ────────────────────────────────────────────────────────────────
// CHART UPDATE FUNCTIONS
// ────────────────────────────────────────────────────────────────
function updateForecastChart(predictions, threshold) {
    const lastHistoricalPoints = historicalData.slice(-20);
    const allLabels = [
        ...lastHistoricalPoints.map((_, i) => `T-${20 - i}`),
        ...predictions.map((_, i) => `T+${i + 1}`),
    ];

    const actualData = lastHistoricalPoints.map(d => d.bandwidth || d.Bandwidth || d[0] || 0);
    const predData = predictions.map(d => d.bandwidth || d.Bandwidth || d[0] || d || 0);

    // Fill nulls for alignment
    const actualSeries = [...actualData, ...new Array(predictions.length).fill(null)];
    const predSeries = [...new Array(lastHistoricalPoints.length - 1).fill(null), actualData[actualData.length - 1], ...predData];

    // Congestion zone — fill values above threshold
    const congestionThreshold = threshold || 650;
    const congestionSeries = allLabels.map(() => congestionThreshold);

    forecastChart.data.labels = allLabels;
    forecastChart.data.datasets[0].data = actualSeries;
    forecastChart.data.datasets[1].data = predSeries;
    forecastChart.data.datasets[2].data = congestionSeries;
    forecastChart.update('active');
}


// ────────────────────────────────────────────────────────────────
// UI HELPER FUNCTIONS
// ────────────────────────────────────────────────────────────────

/**
 * Animate a numeric value from start to end.
 */
function animateValue(element, start, end, duration) {
    if (!element) return;
    const startTime = performance.now();
    const isFloat = end % 1 !== 0;

    function tick(now) {
        const elapsed = now - startTime;
        const progress = Math.min(elapsed / duration, 1);
        // Ease out quad
        const eased = 1 - (1 - progress) * (1 - progress);
        const current = start + (end - start) * eased;
        element.textContent = isFloat ? current.toFixed(1) : Math.round(current);

        if (progress < 1) {
            requestAnimationFrame(tick);
        } else {
            element.textContent = formatNumber(end);
        }
    }

    requestAnimationFrame(tick);
}


/**
 * Update trend indicator arrow and text.
 */
function updateTrend(trendContainer, changePct) {
    if (!trendContainer) return;
    const arrow = trendContainer.querySelector('.trend-arrow');
    const text = trendContainer.querySelector('.trend-text');

    if (changePct >= 0) {
        arrow.textContent = '↑';
        arrow.className = 'trend-arrow trend-up';
    } else {
        arrow.textContent = '↓';
        arrow.className = 'trend-arrow trend-down';
    }

    text.textContent = `${Math.abs(changePct).toFixed(1)}% from previous`;
}


/**
 * Update congestion risk indicator color and label.
 */
function updateCongestionIndicator(probability) {
    const dot = dom.congestionDot();
    const level = dom.congestionLevel();
    const marker = dom.severityMarker();

    let color, label;
    if (probability < CONGESTION_THRESHOLDS.low) {
        color = 'var(--accent-green)';
        label = 'Low Risk';
    } else if (probability < CONGESTION_THRESHOLDS.medium) {
        color = 'var(--accent-yellow)';
        label = 'Medium Risk';
    } else {
        color = 'var(--accent-red)';
        label = 'High Risk';
    }

    if (dot) {
        dot.style.background = color;
    }
    if (level) {
        level.textContent = label;
        level.style.color = color;
    }

    // Update severity bar marker
    if (marker) {
        const percent = Math.min(Math.max(probability * 100, 0), 100);
        marker.style.left = `${percent}%`;
        marker.style.borderColor = color;
    }
}


/**
 * Get recommended action text based on congestion probability.
 */
function getRecommendedAction(probability) {
    if (probability < CONGESTION_THRESHOLDS.low) {
        return 'Normal Operations';
    } else if (probability < CONGESTION_THRESHOLDS.medium) {
        return 'Monitor Closely';
    } else if (probability < CONGESTION_THRESHOLDS.high) {
        return 'Activate QoS Policies';
    } else {
        return 'Immediate Load Balancing';
    }
}


/**
 * Display AI insights in the panel.
 */
function displayInsights(insights) {
    const placeholder = dom.insightsPlaceholder();
    const loading = dom.insightsLoading();
    const result = dom.insightsResult();
    const analysisEl = dom.analysisText();
    const recList = dom.recommendationsList();

    // Hide placeholder & loading, show result
    placeholder.classList.add('hidden');
    loading.classList.add('hidden');
    result.classList.remove('hidden');

    // Render analysis text (basic markdown-style formatting)
    let analysisHtml = '';
    const text = insights.analysis || insights.text || insights;

    if (typeof text === 'string') {
        analysisHtml = text
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\n/g, '<br>')
            .replace(/• /g, '<br>• ');
    }
    analysisEl.innerHTML = analysisHtml;

    // Render recommendations
    recList.innerHTML = '';
    const recommendations = insights.recommendations || [];
    recommendations.forEach(rec => {
        const li = document.createElement('li');
        li.textContent = rec;
        recList.appendChild(li);
    });

    // Update severity bar
    if (insights.severity !== undefined) {
        updateCongestionIndicator(insights.severity);
    }
}


/**
 * Show a toast notification.
 */
function showToast(message, type = 'info') {
    const container = dom.toastContainer();
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;

    const icons = {
        success: '✓',
        error: '✕',
        info: 'ℹ',
    };

    toast.innerHTML = `
        <span class="toast-icon">${icons[type] || icons.info}</span>
        <span class="toast-message">${escapeHtml(message)}</span>
    `;

    container.appendChild(toast);

    // Auto-dismiss after 4 seconds
    setTimeout(() => {
        toast.classList.add('dismissing');
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}


/**
 * Toggle loading state for the generate button and insights panel.
 */
function setLoading(state) {
    isLoading = state;
    const btn = dom.btnGenerate();
    const btnContent = btn.querySelector('.btn-content');
    const btnLoading = btn.querySelector('.btn-loading');

    if (state) {
        btn.disabled = true;
        btnContent.classList.add('hidden');
        btnLoading.classList.remove('hidden');

        // Show skeletons in insights
        dom.insightsPlaceholder().classList.add('hidden');
        dom.insightsResult().classList.add('hidden');
        dom.insightsLoading().classList.remove('hidden');
    } else {
        btn.disabled = false;
        btnContent.classList.remove('hidden');
        btnLoading.classList.add('hidden');
        dom.insightsLoading().classList.add('hidden');
    }
}


/**
 * Format a number with locale-aware separators.
 */
function formatNumber(n) {
    if (typeof n !== 'number' || isNaN(n)) return '—';
    if (n % 1 !== 0) return n.toFixed(1);
    return n.toLocaleString('en-US');
}


/**
 * Escape HTML entities to prevent XSS.
 */
function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}


/**
 * Get severity color based on probability.
 */
function getSeverityColor(probability) {
    if (probability < CONGESTION_THRESHOLDS.low) return 'var(--accent-green)';
    if (probability < CONGESTION_THRESHOLDS.medium) return 'var(--accent-yellow)';
    return 'var(--accent-red)';
}
