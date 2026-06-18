/* ===========================
   CHARTS.JS
   Chart.js visualizations
=========================== */

let eegChart = null;

/* ===========================
   EEG SIGNAL CHART
=========================== */

function renderEEGChart(result) {
    const canvas = document.getElementById("eeg-chart");

    if (!canvas) return;

    if (typeof Chart === "undefined") {
        console.error("Chart.js is not loaded.");
        return;
    }

    const signal =
        result.signal_preview ||
        result.eeg_preview ||
        null;

    const labels =
        signal?.time ||
        Array.from({ length: 100 }, (_, i) => i);

    const values =
        signal?.values ||
        generateMockSignal();

    if (eegChart) {
        eegChart.destroy();
    }

    eegChart = new Chart(canvas, {
        type: "line",

        data: {
            labels: labels,

            datasets: [
                {
                    label: "EEG Signal",
                    data: values,
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.25
                }
            ]
        },

        options: {
            responsive: true,
            maintainAspectRatio: false,

            plugins: {
                legend: {
                    display: true
                },

                tooltip: {
                    mode: "index",
                    intersect: false
                }
            },

            scales: {
                x: {
                    title: {
                        display: true,
                        text: "Time"
                    }
                },

                y: {
                    title: {
                        display: true,
                        text: "Amplitude"
                    }
                }
            }
        }
    });
}

/* ===========================
   BRAINWAVE BANDS
=========================== */

function renderBrainwaveBands(result) {
    const container =
        document.getElementById("brainwave-bands");

    if (!container) return;

    const bands =
        result.brainwave_bands ||
        result.bands ||
        {
            delta: 0.22,
            theta: 0.28,
            alpha: 0.20,
            beta: 0.18,
            gamma: 0.12
        };

    const bandLabels = {
        delta: "Delta",
        theta: "Theta",
        alpha: "Alpha",
        beta: "Beta",
        gamma: "Gamma"
    };

    container.innerHTML = "";

    Object.entries(bands).forEach(([key, value]) => {
        const percent = Math.round(value * 100);

        const row = `
            <div class="band-row">

                <div class="band-label">
                    ${bandLabels[key] || key}
                </div>

                <div class="band-bar">
                    <div
                        class="band-fill"
                        style="width:${percent}%"
                    ></div>
                </div>

                <div class="band-value">
                    ${percent}%
                </div>

            </div>
        `;

        container.innerHTML += row;
    });
}

/* ===========================
   MOCK SIGNAL
=========================== */

function generateMockSignal() {
    return Array.from({ length: 100 }, (_, i) => {
        return Math.sin(i / 5) * 20 + Math.random() * 8;
    });
}