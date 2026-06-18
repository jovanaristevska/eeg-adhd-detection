/* ===========================
   MODELS.JS
   Static model information page
=========================== */

const MODEL_INFO = [
    {
        name: "NeuroGPT",
        type: "EEG Conformer + GPT Transformer",
        score: "0.995",
        accuracy: "94.5%",
        balancedAccuracy: "94.9%",
        aucPr: "0.994",
        tags: [
            "Foundation model",
            "Best AUROC",
            "Pretrained"
        ]
    },

    {
        name: "EEGPT",
        type: "Channel-as-token Transformer",
        score: "0.896",
        accuracy: "78.8%",
        balancedAccuracy: "80.8%",
        aucPr: "0.866",
        tags: [
            "Transformer",
            "Scratch training"
        ]
    },

    {
        name: "EEGNet",
        type: "Compact CNN Baseline",
        score: "0.893",
        accuracy: "67.1%",
        balancedAccuracy: "70.5%",
        aucPr: "0.811",
        tags: [
            "CNN",
            "Lightweight",
            "Baseline"
        ]
    }
];

/* ===========================
   MAIN RENDER
=========================== */

function renderModelsPage() {
    renderModelCards();
    renderModelComparisonTable();
}

/* ===========================
   MODEL CARDS
=========================== */

function renderModelCards() {
    const container =
        document.getElementById("models-grid");

    if (!container) return;

    container.innerHTML = "";

    MODEL_INFO.forEach(model => {

        const tags = model.tags
            .map(tag => `<span class="badge badge-info">${tag}</span>`)
            .join("");

        const card = `
            <div class="model-card">

                <div class="model-card-header">

                    <div>
                        <div class="model-card-title">
                            ${model.name}
                        </div>

                        <div class="model-card-subtitle">
                            ${model.type}
                        </div>
                    </div>

                    <div class="model-score">
                        ${model.score}
                    </div>

                </div>

                <div class="metric-grid">

                    <div class="metric-item">
                        <div class="metric-label">
                            Accuracy
                        </div>
                        <div class="metric-value">
                            ${model.accuracy}
                        </div>
                    </div>

                    <div class="metric-item">
                        <div class="metric-label">
                            Balanced Acc
                        </div>
                        <div class="metric-value">
                            ${model.balancedAccuracy}
                        </div>
                    </div>

                    <div class="metric-item">
                        <div class="metric-label">
                            AUC-PR
                        </div>
                        <div class="metric-value">
                            ${model.aucPr}
                        </div>
                    </div>

                    <div class="metric-item">
                        <div class="metric-label">
                            AUROC
                        </div>
                        <div class="metric-value">
                            ${model.score}
                        </div>
                    </div>

                </div>

                <div class="callout">
                    <div class="callout-title">
                        Model characteristics
                    </div>

                    <div class="callout-text">
                        ${tags}
                    </div>
                </div>

            </div>
        `;

        container.innerHTML += card;
    });
}

/* ===========================
   COMPARISON TABLE
=========================== */

function renderModelComparisonTable() {
    const tbody =
        document.getElementById("model-comparison-body");

    if (!tbody) return;

    tbody.innerHTML = "";

    MODEL_INFO.forEach(model => {

        const row = `
            <tr>
                <td>${model.name}</td>
                <td>${model.type}</td>
                <td>${model.score}</td>
                <td>${model.accuracy}</td>
                <td>${model.balancedAccuracy}</td>
                <td>${model.aucPr}</td>
            </tr>
        `;

        tbody.innerHTML += row;
    });
}