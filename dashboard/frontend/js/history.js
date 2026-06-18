/* ===========================
   HISTORY.JS
   Local browser history
=========================== */

const HISTORY_KEY = "eeg_analysis_history";

/* ===========================
   GET HISTORY
=========================== */

function getHistory() {
    try {
        return JSON.parse(
            localStorage.getItem(HISTORY_KEY)
        ) || [];
    } catch {
        return [];
    }
}

/* ===========================
   SAVE HISTORY
=========================== */

function saveAnalysisToHistory(result, fileName) {
    const history = getHistory();

    const item = {
        id: Date.now(),
        fileName: fileName,
        date: new Date().toLocaleString(),

        prediction:
            result.prediction ||
            result.ensemble_prediction ||
            "Unknown",

        confidence:
            result.confidence ||
            result.ensemble_confidence ||
            0,

        raw: result
    };

    history.unshift(item);

    localStorage.setItem(
        HISTORY_KEY,
        JSON.stringify(history)
    );

    loadHistory();
}

/* ===========================
   LOAD HISTORY
=========================== */

function loadHistory() {
    renderHistoryTable();
    renderHistoryStats();
}

/* ===========================
   RENDER TABLE
=========================== */

function renderHistoryTable() {
    const tbody =
        document.getElementById("history-body");

    if (!tbody) return;

    const history = getHistory();

    if (!history.length) {
        tbody.innerHTML = `
            <tr>
                <td colspan="5">
                    <div class="history-empty">
                        No analyses saved yet.
                    </div>
                </td>
            </tr>
        `;
        return;
    }

    tbody.innerHTML = "";

    history.forEach(item => {

        const row = `
            <tr>

                <td>${item.fileName}</td>

                <td>${item.date}</td>

                <td>
                    <span class="prediction-pill">
                        ${item.prediction}
                    </span>
                </td>

                <td>
                    ${Math.round(item.confidence * 100)}%
                </td>

                <td>
                    <div class="history-actions">

                        <button
                            class="btn btn-secondary"
                            onclick="viewHistoryItem(${item.id})"
                        >
                            View
                        </button>

                        <button
                            class="btn btn-danger"
                            onclick="deleteHistoryItem(${item.id})"
                        >
                            Delete
                        </button>

                    </div>
                </td>

            </tr>
        `;

        tbody.innerHTML += row;
    });
}

/* ===========================
   HISTORY STATS
=========================== */

function renderHistoryStats() {
    const totalEl =
        document.getElementById("history-total");

    const adhdEl =
        document.getElementById("history-adhd");

    const controlEl =
        document.getElementById("history-control");

    const history = getHistory();

    const adhdCount =
        history.filter(item =>
            item.prediction
                .toLowerCase()
                .includes("adhd")
        ).length;

    const controlCount =
        history.filter(item =>
            item.prediction
                .toLowerCase()
                .includes("control")
        ).length;

    if (totalEl) {
        totalEl.textContent = history.length;
    }

    if (adhdEl) {
        adhdEl.textContent = adhdCount;
    }

    if (controlEl) {
        controlEl.textContent = controlCount;
    }
}

/* ===========================
   VIEW HISTORY ITEM
=========================== */

function viewHistoryItem(id) {
    const item =
        getHistory().find(entry => entry.id === id);

    if (!item) return;

    latestResult = item.raw;

    renderPrediction(item.raw);
    renderEEGChart(item.raw);
    renderBrainwaveBands(item.raw);
    renderPerModelResults(item.raw);

    showPage("analyze");

    document.querySelectorAll(".nav-item")
        .forEach(nav =>
            nav.classList.remove("active")
        );

    const analyzeNav =
        document.querySelector(
            '[data-page="analyze"]'
        );

    if (analyzeNav) {
        analyzeNav.classList.add("active");
    }
}

/* ===========================
   DELETE HISTORY ITEM
=========================== */

function deleteHistoryItem(id) {
    const updatedHistory =
        getHistory().filter(item => item.id !== id);

    localStorage.setItem(
        HISTORY_KEY,
        JSON.stringify(updatedHistory)
    );

    loadHistory();

    showToast(
        "Deleted",
        "History item removed.",
        "success"
    );
}

/* ===========================
   CLEAR HISTORY
=========================== */

function clearHistory() {
    localStorage.removeItem(HISTORY_KEY);

    loadHistory();

    showToast(
        "History cleared",
        "All saved analyses were removed.",
        "success"
    );
}

/* ===========================
   EXPORT CSV
=========================== */

function exportHistoryCSV() {
    const history = getHistory();

    if (!history.length) {
        showToast(
            "No history",
            "There is nothing to export.",
            "warning"
        );

        return;
    }

    const rows = [
        ["File", "Date", "Prediction", "Confidence"],

        ...history.map(item => [
            item.fileName,
            item.date,
            item.prediction,
            `${Math.round(item.confidence * 100)}%`
        ])
    ];

    const csv =
        rows.map(row => row.join(",")).join("\n");

    const blob =
        new Blob([csv], { type: "text/csv" });

    const url =
        URL.createObjectURL(blob);

    const link =
        document.createElement("a");

    link.href = url;
    link.download = "eeg-analysis-history.csv";
    link.click();

    URL.revokeObjectURL(url);
}