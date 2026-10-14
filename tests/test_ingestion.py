import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.config import get_config
from src.ingestion.pdf_parser import PDFIngestor
from src.nlp.scientific_ner import ScientificNERExtractor
from src.retrieval.hierarchical_indexer import HierarchicalIndexer
from src.vision.figure_processor import FigureProcessor
from src.grounding.entity_graph import EntityGroundingGraph
from src.generation.caption_generator import GroundedCaptionGenerator
from src.evaluation.hallucination_checker import HallucinationChecker

def test_pdf_ingestion():
    config = get_config()
    ingestor = PDFIngestor(config)
    pdf_path = "demo/sample_paper.pdf"
    if os.path.exists(pdf_path):
        doc = ingestor.parse_pdf(pdf_path)
        assert doc["paper_id"] == "sample_paper"
        assert len(doc["sections"]) > 0
        print("✓ test_pdf_ingestion passed")

def test_scientific_ner():
    config = get_config()
    ner = ScientificNERExtractor(config=config)
    text = "ResNet achieved 85.3% top-1 accuracy on CIFAR-10"
    entities = ner.extract_entities(text)
    assert len(entities) > 0
    entity_texts = [e["text"] for e in entities]
    assert "ResNet" in entity_texts
    assert "CIFAR-10" in entity_texts
    print("✓ test_scientific_ner passed")

def test_hierarchical_retrieval():
    indexer = HierarchicalIndexer()
    doc_struct = {
        "paper_id": "test_doc",
        "title": "Test Paper Title",
        "sections": [
            {
                "section_id": "sec_1",
                "title": "Intro",
                "paragraphs": [{"paragraph_id": "p1", "text": "ResNet-50 achieves 85.3% accuracy."}]
            }
        ]
    }
    indexer.index_document(doc_struct)
    res = indexer.retrieve("accuracy")
    assert "weighted_score" in res
    assert len(res["level_results"]["paper"]) > 0
    print("✓ test_hierarchical_retrieval passed")

def test_figure_processor():
    import pytest
    processor = FigureProcessor()
    if not processor.ocr.available:
        pytest.skip(f"OCR engine unavailable: {processor.ocr.unavailable_reason}")
    res = processor.process_figure("demo/sample_figure.png")
    assert "elements" in res
    texts = " ".join(e["text"] for e in res["elements"]).lower()
    # the demo figure has a real x-axis label "Training Epochs (0 - 300)"
    assert "epoch" in texts
    assert any(e["type"] == "CHART_REGION" for e in res["elements"])
    print("✓ test_figure_processor passed")


def test_figure_processor_missing_file_returns_no_elements():
    res = FigureProcessor().process_figure("does/not/exist.png")
    assert res["elements"] == [] and "NOT AVAILABLE" in res["status"]

def test_entity_grounding_graph():
    graph_builder = EntityGroundingGraph()
    elements = [{"id": "axis_y", "type": "AXIS", "text": "Accuracy"}]
    entities = [{"text": "Accuracy", "type": "METRIC"}]
    graph_builder.build_graph(elements, entities)
    summary = graph_builder.get_grounding_summary()
    assert summary["num_nodes"] > 0
    print("✓ test_entity_grounding_graph passed")

def test_hallucination_checker():
    checker = HallucinationChecker()
    res = checker.verify_caption(
        "ResNet achieved accuracy.",
        [{"text": "ResNet achieved 85.3% accuracy."}]
    )
    assert res["technical_correctness_ratio"] == 1.0
    print("✓ test_hallucination_checker passed")

if __name__ == "__main__":
    test_pdf_ingestion()
    test_scientific_ner()
    test_hierarchical_retrieval()
    test_figure_processor()
    test_entity_grounding_graph()
    test_hallucination_checker()
    print("\nALL BACKEND UNIT TESTS PASSED!")


def test_api_figure_image_endpoint_restricts_paths():
    from fastapi.testclient import TestClient
    from api.main import app
    c = TestClient(app)
    assert c.get("/figure-image", params={"path": "demo/sample_figure.png"}).status_code == 200
    assert c.get("/figure-image", params={"path": "config/config.yaml"}).status_code == 403
    assert c.get("/figure-image", params={"path": "demo/../config/config.yaml"}).status_code == 403
    assert c.get("/figure-image", params={"path": "/etc/passwd"}).status_code == 403
