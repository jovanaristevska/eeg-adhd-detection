#OVA E PRVIOT MAIN SO RABOTEVME SAMO SO 19 KANALI
# import logging
# import os
# import sys
# import tempfile

# from fastapi import FastAPI, File, Form, HTTPException, UploadFile
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import FileResponse
# from fastapi.staticfiles import StaticFiles

# # Make EEG-FM-Bench importable
# BENCH_PATH = 'D:/EEG-FM-Bench'
# if BENCH_PATH not in sys.path:
#     sys.path.insert(0, BENCH_PATH)

# from backend.model_loader import load_eegpt, load_eegnet, load_neurogpt
# from backend.preprocessor import PreprocessorPipeline
# from backend.inference import InferenceEngine

# logging.basicConfig(level=logging.INFO,
#                     format='%(asctime)s %(levelname)s %(name)s: %(message)s')
# logger = logging.getLogger(__name__)

# # ── paths ────────────────────────────────────────────────────────────────────
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# MODELS_DIR = os.path.join(BASE_DIR, 'models')
# FRONTEND_DIR = os.path.join(BASE_DIR, 'frontend')
# EEGPT_PATH = os.path.join(MODELS_DIR, 'eegpt_unified_epoch_6.pt')
# EEGNET_PATH = os.path.join(MODELS_DIR, 'eegnet_adhd_epoch_10.pt')
# NEUROGPT_PATH = os.path.join(MODELS_DIR, 'neurogpt_adhd_epoch_7.pt')

# # ── model loading ─────────────────────────────────────────────────────────────
# logger.info("Loading models...")
# eegpt_model = load_eegpt(EEGPT_PATH) if os.path.exists(EEGPT_PATH) else None
# eegnet_model = load_eegnet(EEGNET_PATH) if os.path.exists(EEGNET_PATH) else None
# neurogpt_model = load_neurogpt(NEUROGPT_PATH) if os.path.exists(NEUROGPT_PATH) else None


# if eegpt_model is None:
#     logger.warning("EEGPT model not loaded — running in MOCK mode for EEGPT")
# if eegnet_model is None:
#     logger.warning("EEGNet model not loaded — running in MOCK mode for EEGNet")
# if neurogpt_model is None:
#     logger.warning("NeuroGPT model not loaded — running in MOCK mode for NeuroGPT")

# MOCK_MODE = eegpt_model is None and eegnet_model is None and neurogpt_model is None

# preprocessor = PreprocessorPipeline()
# engine = InferenceEngine()

# # ── model metadata ────────────────────────────────────────────────────────────
# MODEL_INFO = {
#     'eegpt': {
#         'name': 'EEG-GPT',
#         'accuracy': 0.788,
#         'auroc': 0.896,
#         'balanced_acc': 0.808,
#         'auc_pr': 0.866,
#         'epochs': 11,
#         'best_epoch': 6,
#         'batch_size': 32,
#         'description': 'Foundation model trained from scratch (no pretrained weights)',
#     },
#     'eegnet': {
#         'name': 'EEGNet',
#         'accuracy': 0.671,
#         'auroc': 0.893,
#         'balanced_acc': 0.705,
#         'auc_pr': 0.811,
#         'epochs': 15,
#         'best_epoch': 10,
#         'batch_size': 1024,
#         'description': 'Lightweight CNN baseline for EEG classification',
#     },
#     'neurogpt': {
#         'name': 'NeuroGPT',
#         'accuracy': 0.945,
#         'auroc': 0.995,
#         'balanced_acc': 0.949,
#         'auc_pr': 0.994,
#         'epochs': 12,
#         'best_epoch': 7,
#         'batch_size': 32,
#         'description': 'EEG Conformer + GPT foundation model (pretrained)',
#     },
# }

# # Map model_type -> (loaded model, mock flag)
# def _get_model(model_type: str):
#     if model_type == 'eegpt':
#         return eegpt_model
#     elif model_type == 'eegnet':
#         return eegnet_model
#     elif model_type == 'neurogpt':
#         return neurogpt_model
#     return None


# # ── FastAPI app ───────────────────────────────────────────────────────────────
# app = FastAPI(title='EEG Analysis Dashboard API')

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=['*'],
#     allow_methods=['*'],
#     allow_headers=['*'],
# )

# # Serve frontend at root
# @app.get('/', include_in_schema=False)
# async def serve_frontend():
#     index_path = os.path.join(FRONTEND_DIR, 'index.html')
#     if os.path.exists(index_path):
#         return FileResponse(index_path, media_type='text/html')
#     raise HTTPException(status_code=404, detail='Frontend not found')


# # ── endpoints ─────────────────────────────────────────────────────────────────

# @app.get('/health')
# async def health():
#     return {
#         'status': 'ok',
#         'models_loaded': {
#             'eegpt': eegpt_model is not None,
#             'eegnet': eegnet_model is not None,
#             'neurogpt': neurogpt_model is not None,
#         },
#         'mock_mode': MOCK_MODE,
#     }


# @app.get('/model_info/{model_type}')
# async def model_info(model_type: str):
#     if model_type not in MODEL_INFO:
#         raise HTTPException(status_code=404, detail=f'Unknown model: {model_type}')
#     return MODEL_INFO[model_type]


# @app.post('/predict')
# async def predict(
#     file: UploadFile = File(...),
#     model_type: str = Form(...),
# ):
#     """Single-model prediction. Kept for backward compatibility."""
#     if model_type not in ('eegpt', 'eegnet', 'neurogpt'):
#         raise HTTPException(status_code=400,
#                             detail=f"model_type must be 'eegpt', 'eegnet', or 'neurogpt', got '{model_type}'")

#     # Save upload to a temp file
#     suffix = os.path.splitext(file.filename or '.csv')[1] or '.csv'
#     with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
#         content = await file.read()
#         tmp.write(content)
#         tmp_path = tmp.name

#     try:
#         # Preprocess
#         try:
#             preproc_result = preprocessor.preprocess(tmp_path)
#         except ValueError as exc:
#             raise HTTPException(status_code=422, detail=str(exc))
#         except Exception as exc:
#             logger.exception("Preprocessing failed")
#             raise HTTPException(status_code=500, detail=f'Preprocessing error: {exc}')

#         windows_tensor = preproc_result['windows_tensor']
#         eeg_signal = preproc_result['eeg_signal']
#         band_powers = preproc_result['band_powers']
#         n_windows = preproc_result['n_windows']

#         # Select model
#         model = _get_model(model_type)
#         mock_this = model is None

#         # Run inference
#         try:
#             result = engine.predict(windows_tensor, model, model_type)
#         except Exception as exc:
#             logger.exception("Inference failed")
#             raise HTTPException(status_code=500, detail=f'Inference error: {exc}')

#         return {
#             'prediction': result['prediction'],
#             'confidence': result['confidence'],
#             'reliability': result['reliability'],
#             'window_predictions': result['window_predictions'],
#             'eeg_signal': eeg_signal,
#             'band_powers': band_powers,
#             'model_used': model_type,
#             'windows_analyzed': n_windows,
#             'mock_mode': mock_this,
#         }

#     finally:
#         try:
#             os.unlink(tmp_path)
#         except OSError:
#             pass


# @app.post('/predict_all')
# async def predict_all(file: UploadFile = File(...)):
#     """
#     Run all three models on the same preprocessed signal.
#     Returns per-model results + ensemble (majority vote) decision.
#     """
#     # Save upload to a temp file
#     suffix = os.path.splitext(file.filename or '.csv')[1] or '.csv'
#     with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
#         content = await file.read()
#         tmp.write(content)
#         tmp_path = tmp.name

#     try:
#         # ── 1. Preprocess ONCE ────────────────────────────────────────────────
#         try:
#             preproc_result = preprocessor.preprocess(tmp_path)
#         except ValueError as exc:
#             raise HTTPException(status_code=422, detail=str(exc))
#         except Exception as exc:
#             logger.exception("Preprocessing failed")
#             raise HTTPException(status_code=500, detail=f'Preprocessing error: {exc}')

#         windows_tensor = preproc_result['windows_tensor']
#         eeg_signal = preproc_result['eeg_signal']
#         band_powers = preproc_result['band_powers']
#         n_windows = preproc_result['n_windows']

#         # ── 2. Run inference on each model ────────────────────────────────────
#         per_model = {}
#         for model_type in ('neurogpt', 'eegpt', 'eegnet'):
#             model = _get_model(model_type)
#             mock_this = model is None
#             try:
#                 result = engine.predict(windows_tensor, model, model_type)
#                 per_model[model_type] = {
#                     'prediction': result['prediction'],
#                     'confidence': result['confidence'],
#                     'reliability': result['reliability'],
#                     'window_predictions': result['window_predictions'],
#                     'mock_mode': mock_this,
#                     'model_info': MODEL_INFO.get(model_type, {}),
#                 }
#             except Exception as exc:
#                 logger.exception(f"Inference failed for {model_type}")
#                 per_model[model_type] = {
#                     'prediction': None,
#                     'confidence': 0.0,
#                     'reliability': 'Low',
#                     'window_predictions': [],
#                     'mock_mode': mock_this,
#                     'error': str(exc),
#                     'model_info': MODEL_INFO.get(model_type, {}),
#                 }

#         # ── 3. Compute ensemble (majority vote across models) ─────────────────
#         valid_preds = [m['prediction'] for m in per_model.values() if m['prediction']]
#         n_models = len(valid_preds)

#         if n_models == 0:
#             ensemble_prediction = None
#             ensemble_confidence = 0.0
#             ensemble_reliability = 'Low'
#             agreement_str = '0/0'
#         else:
#             adhd_votes = sum(1 for p in valid_preds if p == 'ADHD')
#             control_votes = n_models - adhd_votes

#             ensemble_prediction = 'ADHD' if adhd_votes >= control_votes else 'Control'

#             # Average confidence ONLY among models that agree with the winner
#             agreeing_confs = [
#                 m['confidence'] for m in per_model.values()
#                 if m['prediction'] == ensemble_prediction
#             ]
#             ensemble_confidence = sum(agreeing_confs) / len(agreeing_confs) if agreeing_confs else 0.0

#             agreeing_count = max(adhd_votes, control_votes)
#             agreement_str = f"{agreeing_count}/{n_models}"

#             # Reliability: full agreement + high conf = High
#             if agreeing_count == n_models and ensemble_confidence >= 80:
#                 ensemble_reliability = 'High'
#             elif agreeing_count == n_models:
#                 ensemble_reliability = 'Medium'
#             elif agreeing_count >= 2:
#                 ensemble_reliability = 'Medium'
#             else:
#                 ensemble_reliability = 'Low'

#         return {
#             'models': per_model,
#             'ensemble': {
#                 'prediction': ensemble_prediction,
#                 'confidence': round(ensemble_confidence, 1),
#                 'reliability': ensemble_reliability,
#                 'agreement': agreement_str,
#                 'n_models': n_models,
#             },
#             'eeg_signal': eeg_signal,
#             'band_powers': band_powers,
#             'windows_analyzed': n_windows,
#         }

#     finally:
#         try:
#             os.unlink(tmp_path)
#         except OSError:
#             pass


# OVA E VTORIOT MAIN SO RABOTEVME I SO 8 KANALI, DODADOVME ROUTING MEGJU 19-CH I 8-CH MODELI BASED NA FORMATOT NA VNEŠNIOT SIGNAL (CLINICAL EEG VS CROWN)
import logging
import os
import sys
import tempfile

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# Make EEG-FM-Bench importable
BENCH_PATH = 'D:/EEG-FM-Bench'
if BENCH_PATH not in sys.path:
    sys.path.insert(0, BENCH_PATH)

from backend.model_loader import load_eegpt, load_eegnet, load_neurogpt
from backend.preprocessor import PreprocessorPipeline
from backend.inference import InferenceEngine

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(name)s: %(message)s')
logger = logging.getLogger(__name__)

# ── paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
FRONTEND_DIR = os.path.join(BASE_DIR, 'frontend')

# 19-channel (Clinical EEG / ADHD dataset)
EEGPT_19_PATH    = os.path.join(MODELS_DIR, 'eegpt_unified_epoch_6.pt')
EEGNET_19_PATH   = os.path.join(MODELS_DIR, 'eegnet_adhd_epoch_10.pt')
NEUROGPT_19_PATH = os.path.join(MODELS_DIR, 'neurogpt_adhd_epoch_7.pt')

# 8-channel (Neurosity Crown / ADHD_Crown dataset)
EEGPT_8_PATH    = os.path.join(MODELS_DIR, 'eegpt_adhd_crown_last.pt')
EEGNET_8_PATH   = os.path.join(MODELS_DIR, 'eegnet_adhd_crown_last.pt')
NEUROGPT_8_PATH = os.path.join(MODELS_DIR, 'neurogpt_adhd_crown_last.pt')


# ── model loading ─────────────────────────────────────────────────────────────
logger.info("Loading 19-channel models (Clinical EEG)...")
models_19ch = {
    'eegpt':    load_eegpt(EEGPT_19_PATH,    n_chans=19) if os.path.exists(EEGPT_19_PATH)    else None,
    'eegnet':   load_eegnet(EEGNET_19_PATH,  n_chans=19) if os.path.exists(EEGNET_19_PATH)   else None,
    'neurogpt': load_neurogpt(NEUROGPT_19_PATH, n_chans=19) if os.path.exists(NEUROGPT_19_PATH) else None,
}

logger.info("Loading 8-channel models (Neurosity Crown)...")
models_8ch = {
    'eegpt':    load_eegpt(EEGPT_8_PATH,    n_chans=8) if os.path.exists(EEGPT_8_PATH)    else None,
    'eegnet':   load_eegnet(EEGNET_8_PATH,  n_chans=8) if os.path.exists(EEGNET_8_PATH)   else None,
    'neurogpt': load_neurogpt(NEUROGPT_8_PATH, n_chans=8) if os.path.exists(NEUROGPT_8_PATH) else None,
}

# Warn about missing models per format
for name, m in models_19ch.items():
    if m is None:
        logger.warning(f"{name.upper()} [19ch] not loaded — MOCK mode for this model on 19-channel input")
for name, m in models_8ch.items():
    if m is None:
        logger.warning(f"{name.upper()} [8ch] not loaded — MOCK mode for this model on 8-channel input")

MOCK_MODE_19 = all(m is None for m in models_19ch.values())
MOCK_MODE_8  = all(m is None for m in models_8ch.values())
MOCK_MODE    = MOCK_MODE_19 and MOCK_MODE_8

preprocessor = PreprocessorPipeline()
engine = InferenceEngine()


# ── model metadata ────────────────────────────────────────────────────────────
# 19-channel (ADHD dataset, Clinical EEG)
MODEL_INFO = {
    'eegpt': {
        'name': 'EEG-GPT',
        'accuracy': 0.788,
        'auroc': 0.896,
        'balanced_acc': 0.808,
        'auc_pr': 0.866,
        'epochs': 11,
        'best_epoch': 6,
        'batch_size': 32,
        'description': 'Foundation model trained from scratch (no pretrained weights)',
    },
    'eegnet': {
        'name': 'EEGNet',
        'accuracy': 0.671,
        'auroc': 0.893,
        'balanced_acc': 0.705,
        'auc_pr': 0.811,
        'epochs': 15,
        'best_epoch': 10,
        'batch_size': 1024,
        'description': 'Lightweight CNN baseline for EEG classification',
    },
    'neurogpt': {
        'name': 'NeuroGPT',
        'accuracy': 0.945,
        'auroc': 0.995,
        'balanced_acc': 0.949,
        'auc_pr': 0.994,
        'epochs': 12,
        'best_epoch': 7,
        'batch_size': 32,
        'description': 'EEG Conformer + GPT foundation model (pretrained)',
    },
}

# 8-channel (ADHD_Crown dataset, Neurosity Crown)
MODEL_INFO_CROWN = {
    'eegpt': {
        'name': 'EEG-GPT (Crown)',
        'accuracy': 0.664,
        'auroc': 0.704,
        'balanced_acc': 0.685,
        'auc_pr': 0.703,
        'epochs': 18,
        'best_epoch': 18,
        'batch_size': 32,
        'description': 'EEG-GPT adapted for Neurosity Crown 8-channel input (10-10 system)',
    },
    'eegnet': {
        'name': 'EEGNet (Crown)',
        'accuracy': 0.665,
        'auroc': 0.700,
        'balanced_acc': 0.685,
        'auc_pr': 0.687,
        'epochs': 15,
        'best_epoch': 15,
        'batch_size': 1024,
        'description': 'Lightweight CNN for Neurosity Crown 8-channel input',
    },
    'neurogpt': {
        'name': 'NeuroGPT (Crown)',
        'accuracy': 0.583,
        'auroc': 0.763,
        'balanced_acc': 0.625,
        'auc_pr': 0.738,
        'epochs': 11,
        'best_epoch': 8,
        'batch_size': 32,
        'description': 'NeuroGPT adapted for Neurosity Crown 8-channel input',
    },
}


def _get_model(model_type: str, format_name: str = 'adhd_19'):
    """Return loaded model for the requested model_type and EEG format."""
    if format_name == 'crown_8':
        return models_8ch.get(model_type)
    return models_19ch.get(model_type)


def _get_model_info(model_type: str, format_name: str = 'adhd_19'):
    """Return MODEL_INFO entry for the requested model_type and EEG format."""
    if format_name == 'crown_8':
        return MODEL_INFO_CROWN.get(model_type, {})
    return MODEL_INFO.get(model_type, {})


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title='EEG Analysis Dashboard API')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)

# Serve frontend at root
@app.get('/', include_in_schema=False)
async def serve_frontend():
    index_path = os.path.join(FRONTEND_DIR, 'index.html')
    if os.path.exists(index_path):
        return FileResponse(index_path, media_type='text/html')
    raise HTTPException(status_code=404, detail='Frontend not found')


# ── endpoints ─────────────────────────────────────────────────────────────────

@app.get('/health')
async def health():
    return {
        'status': 'ok',
        'models_loaded': {
            'adhd_19': {name: m is not None for name, m in models_19ch.items()},
            'crown_8': {name: m is not None for name, m in models_8ch.items()},
        },
        'mock_mode': MOCK_MODE,
    }


@app.get('/model_info/{model_type}')
async def model_info(model_type: str, format: str = 'adhd_19'):
    """
    Get model metrics. Optional query parameter format=crown_8 returns
    the 8-channel Crown variant.
    """
    if model_type not in MODEL_INFO:
        raise HTTPException(status_code=404, detail=f'Unknown model: {model_type}')
    if format == 'crown_8':
        return MODEL_INFO_CROWN.get(model_type, {})
    return MODEL_INFO[model_type]


@app.post('/predict')
async def predict(
    file: UploadFile = File(...),
    model_type: str = Form(...),
):
    """Single-model prediction. Auto-routes between 19-ch and 8-ch models based on input."""
    if model_type not in ('eegpt', 'eegnet', 'neurogpt'):
        raise HTTPException(status_code=400,
                            detail=f"model_type must be 'eegpt', 'eegnet', or 'neurogpt', got '{model_type}'")

    # Save upload to a temp file
    suffix = os.path.splitext(file.filename or '.csv')[1] or '.csv'
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # ── Preprocess (auto-detects format) ─────────────────────────────────
        try:
            preproc_result = preprocessor.preprocess(tmp_path)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:
            logger.exception("Preprocessing failed")
            raise HTTPException(status_code=500, detail=f'Preprocessing error: {exc}')

        windows_tensor = preproc_result['windows_tensor']
        eeg_signal     = preproc_result['eeg_signal']
        band_powers    = preproc_result['band_powers']
        n_windows      = preproc_result['n_windows']
        format_name    = preproc_result['format']         # 'adhd_19' or 'crown_8'
        n_channels     = preproc_result['n_channels']
        channels       = preproc_result['channels']

        # ── Select model based on detected format ────────────────────────────
        model = _get_model(model_type, format_name)
        mock_this = model is None
        if mock_this:
            logger.warning(f"Model {model_type} for format {format_name} not loaded — falling back to mock")

        # ── Run inference ────────────────────────────────────────────────────
        try:
            result = engine.predict(windows_tensor, model, model_type)
        except Exception as exc:
            logger.exception("Inference failed")
            raise HTTPException(status_code=500, detail=f'Inference error: {exc}')

        return {
            'prediction':         result['prediction'],
            'confidence':         result['confidence'],
            'reliability':        result['reliability'],
            'window_predictions': result['window_predictions'],
            'eeg_signal':         eeg_signal,
            'band_powers':        band_powers,
            'model_used':         model_type,
            'windows_analyzed':   n_windows,
            'mock_mode':          mock_this,
            'format':             format_name,
            'n_channels':         n_channels,
            'channels':           channels,
        }

    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@app.post('/predict_all')
async def predict_all(file: UploadFile = File(...)):
    """
    Run all three models on the same preprocessed signal.
    Auto-detects format (19-channel ADHD or 8-channel Crown) and routes
    to the matching model set. Returns per-model results + ensemble decision.
    """
    # Save upload to a temp file
    suffix = os.path.splitext(file.filename or '.csv')[1] or '.csv'
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # ── 1. Preprocess ONCE (auto-detects format) ─────────────────────────
        try:
            preproc_result = preprocessor.preprocess(tmp_path)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:
            logger.exception("Preprocessing failed")
            raise HTTPException(status_code=500, detail=f'Preprocessing error: {exc}')

        windows_tensor = preproc_result['windows_tensor']
        eeg_signal     = preproc_result['eeg_signal']
        band_powers    = preproc_result['band_powers']
        n_windows      = preproc_result['n_windows']
        format_name    = preproc_result['format']         # 'adhd_19' or 'crown_8'
        n_channels     = preproc_result['n_channels']
        channels       = preproc_result['channels']

        logger.info(f"Running ensemble inference on {format_name} input ({n_channels} channels)")

        # ── 2. Run inference on each model (routed by format) ────────────────
        per_model = {}
        for model_type in ('neurogpt', 'eegpt', 'eegnet'):
            model = _get_model(model_type, format_name)
            mock_this = model is None
            try:
                result = engine.predict(windows_tensor, model, model_type)
                per_model[model_type] = {
                    'prediction':         result['prediction'],
                    'confidence':         result['confidence'],
                    'reliability':        result['reliability'],
                    'window_predictions': result['window_predictions'],
                    'mock_mode':          mock_this,
                    'model_info':         _get_model_info(model_type, format_name),
                }
            except Exception as exc:
                logger.exception(f"Inference failed for {model_type} on {format_name}")
                per_model[model_type] = {
                    'prediction':         None,
                    'confidence':         0.0,
                    'reliability':        'Low',
                    'window_predictions': [],
                    'mock_mode':          mock_this,
                    'error':              str(exc),
                    'model_info':         _get_model_info(model_type, format_name),
                }

        # ── 3. Compute ensemble (majority vote across models) ────────────────
        valid_preds = [m['prediction'] for m in per_model.values() if m['prediction']]
        n_models = len(valid_preds)

        if n_models == 0:
            ensemble_prediction = None
            ensemble_confidence = 0.0
            ensemble_reliability = 'Low'
            agreement_str = '0/0'
        else:
            adhd_votes = sum(1 for p in valid_preds if p == 'ADHD')
            control_votes = n_models - adhd_votes

            ensemble_prediction = 'ADHD' if adhd_votes >= control_votes else 'Control'

            # Average confidence ONLY among models that agree with the winner
            agreeing_confs = [
                m['confidence'] for m in per_model.values()
                if m['prediction'] == ensemble_prediction
            ]
            ensemble_confidence = sum(agreeing_confs) / len(agreeing_confs) if agreeing_confs else 0.0

            agreeing_count = max(adhd_votes, control_votes)
            agreement_str = f"{agreeing_count}/{n_models}"

            # Reliability: full agreement + high conf = High
            if agreeing_count == n_models and ensemble_confidence >= 80:
                ensemble_reliability = 'High'
            elif agreeing_count == n_models:
                ensemble_reliability = 'Medium'
            elif agreeing_count >= 2:
                ensemble_reliability = 'Medium'
            else:
                ensemble_reliability = 'Low'

        return {
            'models': per_model,
            'ensemble': {
                'prediction':  ensemble_prediction,
                'confidence':  round(ensemble_confidence, 1),
                'reliability': ensemble_reliability,
                'agreement':   agreement_str,
                'n_models':    n_models,
            },
            'eeg_signal':       eeg_signal,
            'band_powers':      band_powers,
            'windows_analyzed': n_windows,
            'format':           format_name,
            'n_channels':       n_channels,
            'channels':         channels,
        }

    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass