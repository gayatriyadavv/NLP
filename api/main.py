"""
FastAPI Backend Server for Multi-Modal Scientific Visualization Understanding
Includes CORS middleware, file upload endpoints, SciCap Dataset loader, Phase 3 PyG GNN,
and Phase 4 3D Visualization Branch (PointNet + Multi-View + Multimodal Retriever).
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Body
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import os
import sys
import shutil
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.config import get_config
from src.ingestion.pdf_parser import PDFIngestor
from src.ingestion.scicap_loader import SciCapDatasetLoader
from src.nlp.scientific_ner import ScientificNERExtractor
from src.retrieval.hierarchical_indexer import HierarchicalIndexer
from src.vision.figure_processor import FigureProcessor
from src.grounding.entity_graph import EntityGroundingGraph
from src.generation.caption_generator import GroundedCaptionGenerator
from src.evaluation.hallucination_checker import HallucinationChecker
from src.vision.threed_detector import ThreeDVisualizationDetector
from src.models.vision.pointnet_extractor import PointNetExtractor, load_point_cloud_fixture
from src.vision.multiview_renderer import MultiViewRenderer
from src.retrieval.multimodal_retriever import MultimodalRetriever
from pipelines.run_pipeline import run_end_to_end_pipeline
import json
import torch

app = FastAPI(
    title="Multi-Modal Scientific Visualization Understanding API",
    version="0.1.0-mvp",
    description="Backend API for PDF ingestion, SciCap benchmark, PyG GNN, 3D PointNet, Multi-View 3D rendering, and multimodal retrieval."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    paper_path: str = "demo/sample_paper.pdf"
    figure_path: str = "demo/sample_figure.png"
    query: str = "What does this figure show regarding accuracy and training epochs?"

class ThreeDDetectRequest(BaseModel):
    figure_path: str = "demo/dataset/samples/scicap_003.png"
    caption_text: Optional[str] = "3D point cloud protein structure"
    paper_context: Optional[str] = "AlphaFold predicted 3D atomic coordinates"

scicap_loader = SciCapDatasetLoader()
threed_detector = ThreeDVisualizationDetector()
pointnet_extractor = PointNetExtractor()
multiview_renderer = MultiViewRenderer()
multimodal_retriever = MultimodalRetriever()

# Trained 3D encoders (produced by pipelines/evaluate_phase4.py); untrained if absent.
ALIGNER_CHECKPOINT = "models/checkpoints/cross_modal_3d.pt"
aligner = None
if os.path.exists(ALIGNER_CHECKPOINT):
    from src.models.vision.cross_modal_3d import CrossModal3DAligner
    aligner = CrossModal3DAligner()
    aligner.load_state_dict(torch.load(ALIGNER_CHECKPOINT, map_location="cpu", weights_only=False)["state_dict"])
    aligner.eval()
    pointnet_extractor = aligner.point_encoder
    multiview_renderer = aligner.view_encoder
WEIGHTS_STATUS = "trained (cross_modal_3d.pt)" if aligner is not None else "UNTRAINED (random init; run pipelines/evaluate_phase4.py)"

@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "Multi-Modal Scientific Visualization Understanding API",
        "version": "0.1.0-mvp",
        "device": get_config().get("system.device", "mps"),
        "phase_status": {
            "phase_1_core_pipeline": "IMPLEMENTED (see implementation_plan.md for per-component status)",
            "phase_2_clip": "BLOCKED (weights not downloadable in build environment)",
            "phase_3_gnn": "IMPLEMENTED (supervised PyG grounding; see results/gnn_grounding_results.json)",
            "phase_4_3d": "PARTIALLY IMPLEMENTED (see implementation_plan.md)",
        },
        "3d_encoder_weights": WEIGHTS_STATUS
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "api", "3d_branch_weights": WEIGHTS_STATUS,
            "3d_detector": "image classifier" if threed_detector.model is not None else "text cues only (no checkpoint)"}

@app.post("/detect-3d")
def detect_3d(req: ThreeDDetectRequest):
    try:
        res = threed_detector.detect_3d_content(req.figure_path, req.caption_text or "", req.paper_context or "")
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/extract-pointcloud-3d")
def extract_pointcloud_3d(filepath: str = "demo/dataset/3d_samples/alphafold_t1087_3d.json"):
    try:
        pts = load_point_cloud_fixture(filepath)
        z_3d = pointnet_extractor.extract_point_cloud_embedding(pts)
        return {
            "num_points": len(pts),
            "embedding_dim": len(z_3d),
            "point_cloud_embedding": z_3d.tolist(),
            "encoder_weights": WEIGHTS_STATUS,
            "status": "IMPLEMENTED (PointNet 3D Feature Extractor)"
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/scicap/samples")
def list_scicap_samples():
    return {
        "dataset_name": scicap_loader.metadata.get("dataset_name"),
        "samples": scicap_loader.list_samples()
    }

@app.get("/scicap/sample/{sample_id}")
def get_scicap_sample(sample_id: str):
    sample = scicap_loader.get_sample(sample_id)
    if not sample:
        raise HTTPException(status_code=404, detail=f"SciCap sample '{sample_id}' not found.")
    doc_struct = scicap_loader.convert_sample_to_document_structure(sample)
    return {"sample": sample, "document_structure": doc_struct}

@app.post("/scicap/run-sample/{sample_id}")
def run_scicap_sample(sample_id: str):
    sample = scicap_loader.get_sample(sample_id)
    if not sample:
        raise HTTPException(status_code=404, detail=f"SciCap sample '{sample_id}' not found.")
    
    fig_path = os.path.join(scicap_loader.dataset_dir, "samples", sample["image_filename"])
    query = f"Explain {sample['sample_id']} figure demonstrating {sample['paper_title']}"
    
    # 3D Detection
    det_3d = threed_detector.detect_3d_content(fig_path, sample["ground_truth_caption"], sample["paragraph_text"])
    
    # Run end-to-end pipeline on this sample's own document text
    doc_struct = scicap_loader.convert_sample_to_document_structure(sample)
    pipeline_res = run_end_to_end_pipeline("", fig_path, query, doc_structure=doc_struct,
                                           caption_text=sample["ground_truth_caption"])

    # Add 3D branch analysis. A figure image carries no raw geometry, so only the
    # image-derived 3D-aligned embedding is available here.
    if det_3d["is_3d"]:
        branch = {"3d_detection": det_3d, "raw_geometry_available": False}
        if aligner is not None:
            from src.models.vision.figure_image_encoder import load_figure_tensor
            with torch.no_grad():
                z_img3d = aligner.encode_image(load_figure_tensor(fig_path).unsqueeze(0))[0].numpy()
            branch["image_to_3d_embedding_dim"] = int(len(z_img3d))
            branch["status"] = "IMPLEMENTED (image encoder aligned to PointNet/multi-view space)"
        else:
            branch["status"] = "PARTIAL (3D detected; aligner checkpoint missing)"
        pipeline_res["3d_branch_analysis"] = branch
    else:
        pipeline_res["3d_branch_analysis"] = {
            "3d_detection": det_3d,
            "status": "NOT APPLICABLE (2D Figure detected)"
        }

    # Evaluate against SciCap Ground Truth
    eval_res = scicap_loader.evaluate_pipeline_output(sample_id, pipeline_res)
    pipeline_res["scicap_evaluation"] = eval_res
    
    return pipeline_res

@app.post("/run-pipeline")
def api_run_pipeline(req: QueryRequest):
    try:
        res = run_end_to_end_pipeline(req.paper_path, req.figure_path, req.query)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/benchmark/results")
def benchmark_results():
    """Measured results files produced by the evaluation pipelines (empty entries if not yet run)."""
    out = {}
    for name in ("gnn_grounding_results", "phase4_results", "ablation_results"):
        path = os.path.join("results", f"{name}.json")
        out[name] = json.load(open(path)) if os.path.exists(path) else None
    return out


# Figure images the UI may display. Only files under these directories are served.
_IMAGE_ROOTS = [os.path.realpath(p) for p in ("demo", os.path.join("data", "benchmark", "images"))]
_IMAGE_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}


@app.get("/figure-image")
def figure_image(path: str):
    """Serves a figure image from the demo / benchmark directories (used by the analysis view)."""
    real = os.path.realpath(path)
    ext = os.path.splitext(real)[1].lower()
    if ext not in _IMAGE_TYPES or not any(real.startswith(root + os.sep) for root in _IMAGE_ROOTS):
        raise HTTPException(status_code=403, detail="Path not allowed")
    if not os.path.isfile(real):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(real, media_type=_IMAGE_TYPES[ext])
