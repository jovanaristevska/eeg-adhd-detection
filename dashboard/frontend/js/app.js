/* ===========================
   APP.JS
=========================== */

const API_BASE = "http://localhost:8000";

let selectedFile = null;
let latestResult = null;

document.addEventListener("DOMContentLoaded", () => {
    initApp();
});

function initApp() {
    initNavigation();
    initUpload();
    initAnalyzeButton();

    loadHistory();
    renderModelsPage();
}

/* ===========================
   NAVIGATION
=========================== */

function initNavigation() {
    document.querySelectorAll(".nav-item").forEach(item => {
        item.addEventListener("click", () => {

            const page = item.dataset.page;

            document.querySelectorAll(".nav-item")
                .forEach(nav => nav.classList.remove("active"));

            item.classList.add("active");

            showPage(page);
        });
    });
}

function showPage(pageName) {

    document.querySelectorAll(".page")
        .forEach(page => page.classList.remove("active"));

    const page = document.getElementById(`${pageName}-page`);

    if(page){
        page.classList.add("active");
    }
}

/* ===========================
   FILE UPLOAD
=========================== */

function initUpload() {

    const uploadZone = document.getElementById("upload-zone");
    const fileInput = document.getElementById("file-input");

    if(!uploadZone || !fileInput) return;

    uploadZone.addEventListener("click", () => {
        fileInput.click();
    });

    fileInput.addEventListener("change", (e) => {

        const file = e.target.files[0];

        if(!file) return;

        selectedFile = file;

        uploadZone.classList.add("file-selected");

        const fileName = document.getElementById("file-name");

        if(fileName){
            fileName.textContent = file.name;
        }

        const analyzeBtn =
            document.getElementById("analyze-btn");

        if(analyzeBtn){
            analyzeBtn.disabled = false;
        }
    });

    uploadZone.addEventListener("dragover", (e)=>{
        e.preventDefault();
        uploadZone.classList.add("dragover");
    });

    uploadZone.addEventListener("dragleave", ()=>{
        uploadZone.classList.remove("dragover");
    });

    uploadZone.addEventListener("drop", (e)=>{
        e.preventDefault();

        uploadZone.classList.remove("dragover");

        const file = e.dataTransfer.files[0];

        if(!file) return;

        selectedFile = file;

        document.getElementById("file-name")
            .textContent = file.name;
    });
}

/* ===========================
   ANALYZE
=========================== */

function initAnalyzeButton() {

    const btn = document.getElementById("analyze-btn");

    if(!btn) return;

    btn.addEventListener("click", analyzeEEG);
}

async function analyzeEEG() {

    if(!selectedFile){
        showToast(
            "Warning",
            "Please upload EEG file first",
            "warning"
        );
        return;
    }

    try{

        showLoading();

        const formData = new FormData();
        formData.append("file", selectedFile);

        const response = await fetch(
            `${API_BASE}/predict`,
            {
                method:"POST",
                body:formData
            }
        );

        if(!response.ok){
            throw new Error("Prediction failed");
        }

        const result = await response.json();

        latestResult = result;

        renderPrediction(result);

        renderEEGChart(result);

        renderBrainwaveBands(result);

        renderPerModelResults(result);

        saveAnalysisToHistory(
            result,
            selectedFile.name
        );

        hideLoading();

        showToast(
            "Success",
            "Analysis completed",
            "success"
        );

    }
    catch(error){

        hideLoading();

        showToast(
            "Error",
            error.message,
            "error"
        );
    }
}

/* ===========================
   RESULTS
=========================== */

function renderPrediction(result){

    const section =
        document.getElementById("results-section");

    if(section){
        section.style.display = "block";
    }

    const prediction =
        result.prediction ||
        result.ensemble_prediction ||
        "Unknown";

    const confidence =
        result.confidence ||
        result.ensemble_confidence ||
        0;

    const label =
        document.getElementById("prediction-label");

    const confidenceEl =
        document.getElementById("confidence-value");

    if(label){
        label.textContent = prediction;
    }

    if(confidenceEl){
        confidenceEl.textContent =
            `${Math.round(confidence*100)}%`;
    }
}

function renderPerModelResults(result){

    const container =
        document.getElementById(
            "per-model-results"
        );

    if(!container) return;

    const models =
        result.models ||
        result.model_results ||
        [];

    container.innerHTML = "";

    models.forEach(model=>{

        const card = `
            <div class="per-model-card">

                <div class="per-model-header">

                    <div>

                        <div class="per-model-title">
                            ${model.name}
                        </div>

                        <div class="model-card-subtitle">
                            ${model.type || ""}
                        </div>

                    </div>

                    <span class="prediction-pill">
                        ${model.prediction}
                    </span>

                </div>

                <div class="model-score">
                    ${Math.round(
                        model.confidence * 100
                    )}%
                </div>

            </div>
        `;

        container.innerHTML += card;
    });
}

/* ===========================
   LOADING
=========================== */

function showLoading(){

    const loading =
        document.getElementById("loading");

    if(loading){
        loading.style.display = "flex";
    }
}

function hideLoading(){

    const loading =
        document.getElementById("loading");

    if(loading){
        loading.style.display = "none";
    }
}

/* ===========================
   TOAST
=========================== */

function showToast(
    title,
    message,
    type="success"
){

    const oldToast =
        document.querySelector(".toast");

    if(oldToast){
        oldToast.remove();
    }

    const toast =
        document.createElement("div");

    toast.className =
        `toast ${type}`;

    toast.innerHTML = `
        <div class="toast-title">
            ${title}
        </div>

        <div class="toast-text">
            ${message}
        </div>
    `;

    document.body.appendChild(toast);

    setTimeout(()=>{
        toast.remove();
    },3000);
}