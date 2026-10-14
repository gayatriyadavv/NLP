"""
Grounded Caption Generation Module
Generates scientific figure captions that only state content backed by available evidence:
extracted figure elements (axes / legends), scientific entities, the grounding graph, and
retrieved paper context.

Status: PROTOTYPE (evidence-slot template). No vision-language model is loaded; the configured
``model_name`` is recorded for reference only. The generator never asserts trends or results
(e.g. "performance gains") that are not present in the retrieved evidence.
"""

from typing import Dict, Any, List


class GroundedCaptionGenerator:
    def __init__(self, model_name: str = "Salesforce/blip-image-captioning-base", config: Dict[str, Any] = None):
        self.model_name = model_name
        self.config = config or {}

    @staticmethod
    def _grounded_entity_texts(graph_summary: Dict[str, Any]) -> set:
        """Entity labels that have at least one edge from a figure element."""
        grounded = set()
        for e in graph_summary.get("edges", []):
            if str(e.get("source", "")).startswith("elem:") and str(e.get("target", "")).startswith("ent:"):
                grounded.add(str(e["target"])[4:].lower())
        return grounded

    def generate_caption(
        self,
        figure_elements: Dict[str, Any],
        retrieved_context: Dict[str, Any],
        entities: List[Dict[str, Any]],
        graph_summary: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generates a technically grounded caption given figure elements, 4-level retrieved context, and entity grounding.
        """
        level_res = retrieved_context.get("level_results", {})
        sec_ctx = (level_res.get("section") or [{}])[0].get("text", "")
        para_ctx = (level_res.get("paragraph") or [{}])[0].get("text", "")

        seen_ents = set()
        dedup_entities = []
        for e in entities:
            if e["text"].lower() not in seen_ents:
                seen_ents.add(e["text"].lower())
                dedup_entities.append(e)
        ent_names = [e["text"] for e in dedup_entities]

        elements = figure_elements.get("elements", [])
        axes = [e["text"] for e in elements if e.get("type") == "AXIS" and e.get("text")]
        legends = [e["text"] for e in elements if e.get("type") == "LEGEND" and e.get("text")]

        methods = [e["text"] for e in dedup_entities if e["type"] in ["METHOD", "MODEL"]]
        metrics = [e["text"] for e in dedup_entities if e["type"] in ["METRIC"]]
        datasets = [e["text"] for e in dedup_entities if e["type"] == "DATASET"]

        sentences = []
        if methods:
            s = f"Comparison of {', '.join(methods[:3])}"
            if metrics:
                s += f" in terms of {', '.join(metrics[:2])}"
            if datasets:
                s += f" on {', '.join(datasets[:2])}"
            sentences.append(s)
        elif metrics:
            sentences.append(f"{', '.join(metrics[:2])} reported" + (f" on {', '.join(datasets[:2])}" if datasets else ""))
        if len(axes) >= 2:
            sentences.append(f"The axes show {axes[0]} and {axes[1]}")
        elif axes:
            sentences.append(f"The axis shows {axes[0]}")
        if legends:
            sentences.append(f"Legend entries: {', '.join(legends[:4])}")
        # The retrieved paragraph is returned as supporting evidence, not copied into the
        # caption: quoting it would make the verifier trivially mark that sentence SUPPORTED.
        if not sentences:
            sentences.append("Insufficient evidence to describe this figure")

        generated_caption = ". ".join(sentences) + "."

        # Fraction of mentioned entities that are grounded to at least one figure element.
        grounded = self._grounded_entity_texts(graph_summary)
        mentioned = [m for m in methods[:3] + metrics[:2] + datasets[:2]]
        grounding_confidence = (
            sum(1 for m in mentioned if m.lower() in grounded) / len(mentioned) if mentioned else 0.0
        )

        supporting_evidence = [
            {"source": "paragraph_level", "text": para_ctx[:150]},
            {"source": "section_level", "text": sec_ctx[:150]},
            {"source": "grounding_graph", "num_grounded_edges": graph_summary.get("num_edges", 0)}
        ]

        return {
            "generated_caption": generated_caption,
            "prompt_used": f"Entities: {', '.join(ent_names)} | Context: {para_ctx[:100]}",
            "entities_used": ent_names,
            "supporting_evidence": supporting_evidence,
            "grounding_confidence": float(round(grounding_confidence, 4)),
            "status": "PROTOTYPE (evidence-slot template; no VLM loaded)"
        }
