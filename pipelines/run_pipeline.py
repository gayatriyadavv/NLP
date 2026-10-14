"""
Main MVP Pipeline Execution Script
Runs end-to-end scientific figure understanding:
1. Ingest PDF & extract document hierarchy
2. Extract Scientific Entities (NER)
3. Build 4-level Hierarchical Index & Retrieve context
4. Process Figure & extract layout/OCR elements
5. Construct Entity Grounding Graph (NetworkX)
6. Generate Grounded Scientific Caption
7. Verify Evidence & Hallucination status
8. Compute SCQ Composite Score
"""

import sys
import os
import argparse
import json

# Add root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.config import get_config
from src.ingestion.pdf_parser import PDFIngestor
from src.nlp.scientific_ner import ScientificNERExtractor
from src.retrieval.hierarchical_indexer import HierarchicalIndexer
from src.vision.figure_processor import FigureProcessor
from src.grounding.entity_graph import EntityGroundingGraph
from src.generation.caption_generator import GroundedCaptionGenerator
from src.evaluation.hallucination_checker import HallucinationChecker
from src.vision.threed_detector import ThreeDVisualizationDetector


def figure_text_reconstruction(caption: str, elements) -> float:
    """
    Fraction of text-bearing figure elements (axes, legends, labels) whose text is reflected in
    the caption. Used as the SCQ 'reconstruction fidelity' term when no reference caption exists.
    """
    texts = [e["text"].lower() for e in elements if e.get("type") in ("AXIS", "LEGEND", "LABEL") and e.get("text")]
    if not texts:
        return 0.0
    cap = caption.lower()
    return sum(1 for t in texts if t in cap) / len(texts)


def run_end_to_end_pipeline(paper_path: str, figure_path: str, query: str, config_path: str = "config/config.yaml",
                            doc_structure=None, caption_text: str = ""):
    print("=" * 60)
    print("Multi-Modal Scientific Visualization Understanding Pipeline")
    print("=" * 60)
    
    config = get_config(config_path)
    
    # 1. Document Ingestion
    print("\n[Step 1/8] Ingesting Scientific Document...")
    ingestor = PDFIngestor(config)
    if doc_structure is not None:
        pass
    elif os.path.exists(paper_path):
        doc_structure = ingestor.parse_pdf(paper_path)
    else:
        print(f"Warning: PDF path '{paper_path}' not found. Using synthetic document structure for prototype demo.")
        doc_structure = {
            "paper_id": "sample_paper_01",
            "title": "Deep Residual Learning for Image Recognition",
            "sections": [
                {
                    "section_id": "sec_1",
                    "title": "1. Introduction",
                    "paragraphs": [
                        {
                            "paragraph_id": "p_1",
                            "text": "Deeper neural networks are more difficult to train. We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously. ResNet achieved 85.3% accuracy on ImageNet dataset."
                        }
                    ]
                }
            ],
            "figures": [],
            "metadata": {"status": "SYNTHETIC FALLBACK (PDF not found; built-in demo text used)"}
        }

    print(f"  -> Extracted Paper Title: '{doc_structure.get('title')}' ({len(doc_structure.get('sections', []))} sections)")

    # 2. Scientific NER
    print("\n[Step 2/8] Extracting Scientific Named Entities...")
    ner = ScientificNERExtractor(config=config)
    full_text = " ".join([p["text"] for sec in doc_structure["sections"] for p in sec["paragraphs"]])
    entities = ner.extract_entities(full_text)
    relations = ner.extract_relations(entities, full_text)
    print(f"  -> Extracted {len(entities)} scientific entities and {len(relations)} relations.")
    for e in entities[:5]:
        print(f"     * [{e['type']}] {e['text']}")

    # 3. Hierarchical Retrieval & Indexing
    print("\n[Step 3/8] Building 4-Level Hierarchical Vector Index & Retrieving Context...")
    indexer = HierarchicalIndexer(weights=config.get("hierarchical_retrieval.weights"))
    indexer.index_document(doc_structure, entities)
    retrieved_context = indexer.retrieve(query, top_k=3)
    print(f"  -> Context Weighted Score for query '{query}': {retrieved_context['weighted_score']:.4f}")

    # 4. Figure Processing
    print("\n[Step 4/8] Processing Figure & Detecting Visual Elements...")
    fig_processor = FigureProcessor(config)
    figure_elements = fig_processor.process_figure(figure_path)
    print(f"  -> Detected {len(figure_elements.get('elements', []))} figure elements (Axes, Legends, Chart Regions).")

    # 5. Entity Grounding Graph
    print("\n[Step 5/8] Constructing Entity Grounding Graph...")
    graph_builder = EntityGroundingGraph(config)
    graph_builder.build_graph(figure_elements.get("elements", []), entities, relations, paragraph=full_text)
    graph_summary = graph_builder.get_grounding_summary()
    print(f"  -> Grounding Graph created: {graph_summary['num_nodes']} nodes, {graph_summary['num_edges']} edges.")

    # 6. Grounded Caption Generation
    print("\n[Step 6/8] Generating Technically Grounded Caption...")
    caption_gen = GroundedCaptionGenerator(config=config)
    caption_output = caption_gen.generate_caption(figure_elements, retrieved_context, entities, graph_summary)
    print(f"\n  [GENERATED CAPTION]:\n  \"{caption_output['generated_caption']}\"\n")

    # 7. Hallucination & Fact Verification
    print("[Step 7/8] Verifying Evidence & Technical Correctness...")
    verifier = HallucinationChecker(config)
    verification_output = verifier.verify_caption(
        caption_output["generated_caption"],
        retrieved_context["level_results"].get("paragraph", [])
    )
    print(f"  -> Technical Correctness Ratio: {verification_output['technical_correctness_ratio']*100:.1f}%")

    # 8. SCQ Metric Evaluation
    print("\n[Step 8/8] Computing SCQ Composite Score...")
    scq_res = verifier.compute_scq_metric(
        entity_coverage=min(len(entities)/5.0, 1.0),
        technical_correctness=verification_output['technical_correctness_ratio'],
        reconstruction_fidelity=figure_text_reconstruction(caption_output["generated_caption"],
                                                           figure_elements.get("elements", [])),
        retrieval_accuracy=retrieved_context['weighted_score']
    )
    print(f"  -> Composite SCQ Score: {scq_res['scq_composite_score']:.4f}")

    # 3D branch: detection from pixels (+ text cues reported separately)
    threed = ThreeDVisualizationDetector(config).detect_3d_content(figure_path, caption_text, full_text)

    # Summary JSON Output
    output_summary = {
        "paper_id": doc_structure["paper_id"],
        "figure_id": figure_elements["figure_id"],
        "figure_path": figure_path,
        "document_structure": doc_structure,
        "query": query,
        "extracted_entities": entities,
        "retrieved_context": retrieved_context,
        "figure_elements": figure_elements,
        "grounding_graph_summary": graph_summary,
        "generated_caption": caption_output["generated_caption"],
        "verification_results": verification_output,
        "scq_evaluation": scq_res,
        "threed_detection": threed,
        "status_declarations": {
            "pdf_ingestion": doc_structure.get("metadata", {}).get("status", "IMPLEMENTED"),
            "scientific_ner": "PROTOTYPE (regex vocabulary; SciBERT not loaded)",
            "hierarchical_retrieval": ("IMPLEMENTED (4-level Sentence-Transformers + FAISS)" if indexer.use_st
                                       else "PARTIAL (4-level word-overlap fallback; sentence-transformers unavailable)"),
            "figure_processing": figure_elements.get("status", "IMPLEMENTED"),
            "entity_grounding_graph": graph_summary.get("status", "IMPLEMENTED"),
            "grounded_captioning": caption_output.get("status", "IMPLEMENTED"),
            "hallucination_verification": verification_output.get("status", "IMPLEMENTED"),
            "cross_modal_clip": "BLOCKED (CLIP weights not downloadable in this environment)",
            "3d_visualization_pathway": threed.get("status", "")
        }
    }

    return output_summary

def main():
    parser = argparse.ArgumentParser(description="Multi-Modal Scientific Visualization Understanding MVP Pipeline")
    parser.add_argument("--paper", type=str, default="sample.pdf", help="Path to input PDF paper")
    parser.add_argument("--figure", type=str, default="figure.png", help="Path to input scientific figure")
    parser.add_argument("--query", type=str, default="What does this figure show?", help="Natural language query")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to YAML configuration")
    parser.add_argument("--output", type=str, default=None, help="Optional output JSON path")

    args = parser.parse_args()
    
    result = run_end_to_end_pipeline(args.paper, args.figure, args.query, args.config)
    
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to: {args.output}")

if __name__ == "__main__":
    main()
