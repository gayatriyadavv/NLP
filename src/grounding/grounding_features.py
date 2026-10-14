"""
Pairwise element <-> entity features and grounding-graph construction used by the supervised
grounding models (GNN and the no-message-passing diagnostic).

All features are deterministic functions of the element text, the entity text/type and the
paragraph text. No gold labels are used here.
"""

import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

TEXT_ELEMENT_TYPES = ("AXIS", "LEGEND", "LABEL")

PAIR_FEATURE_NAMES = [
    "char_trigram_jaccard",
    "difflib_ratio",
    "acronym_match",
    "substring_match",
    "token_overlap",
    "defined_in_paragraph",
    "type_compat_axis_metric_variable",
    "type_compat_legend_label_method",
    "type_compat_label_material_variable",
    "rule_edge",
]


def _norm(t: str) -> str:
    return re.sub(r"[^a-z0-9+]", "", t.lower())


def _strip_units(t: str) -> str:
    return re.sub(r"\(.*?\)", "", t).strip()


def _trigrams(t: str) -> set:
    t = f"  {_norm(t)} "
    return {t[i:i + 3] for i in range(len(t) - 2)}


def _initials(entity: str) -> str:
    words = re.split(r"[\s\-]+", entity)
    return "".join(w[0] for w in words if w and w[0].isalnum()).lower()


def acronym_match(element_text: str, entity_text: str) -> float:
    core = _norm(_strip_units(element_text).split("/")[0])
    ini = _initials(entity_text)
    if not core or len(ini) < 2:
        return 0.0
    return float(core == ini or core.startswith(ini) or ini.startswith(core) and len(core) >= 2)


def defined_in_paragraph(element_text: str, entity_text: str, paragraph: str) -> float:
    """1 if the paragraph contains 'Entity (Alias)' with Alias matching the element text."""
    core = _norm(_strip_units(element_text))
    if not core:
        return 0.0
    for m in re.finditer(re.escape(entity_text) + r"\s*\(([^)]+)\)", paragraph):
        if _norm(m.group(1)) == core:
            return 1.0
    return 0.0


def pair_features(element: Dict[str, Any], entity: Dict[str, Any], paragraph: str,
                  rule_edge: bool) -> np.ndarray:
    et, nt = element.get("text", ""), entity.get("text", "")
    a, b = _trigrams(et), _trigrams(nt)
    jacc = len(a & b) / max(len(a | b), 1)
    ratio = SequenceMatcher(None, _norm(et), _norm(nt)).ratio() if et and nt else 0.0
    ne, nn = _norm(_strip_units(et)), _norm(nt)
    substr = float(bool(ne) and bool(nn) and (ne in nn or nn in ne))
    tok_e = set(re.findall(r"[a-z0-9]+", et.lower()))
    tok_n = set(re.findall(r"[a-z0-9]+", nt.lower()))
    tok = len(tok_e & tok_n) / max(len(tok_n), 1)
    etype, ntype = element.get("type"), entity.get("type")
    return np.array([
        jacc, ratio, acronym_match(et, nt), substr, tok,
        defined_in_paragraph(et, nt, paragraph),
        float(etype == "AXIS" and ntype in ("METRIC", "VARIABLE")),
        float(etype in ("LEGEND", "LABEL") and ntype in ("METHOD", "MODEL")),
        float(etype == "LABEL" and ntype in ("MATERIAL", "VARIABLE", "METRIC")),
        float(rule_edge),
    ], dtype=np.float32)


def build_grounding_graph(elements: List[Dict[str, Any]], entities: List[Dict[str, Any]],
                          relations: Optional[List[Dict[str, Any]]] = None,
                          extra_nodes: Optional[List[Tuple[str, Dict[str, Any], List[str]]]] = None
                          ) -> nx.DiGraph:
    """
    Rule-based NetworkX grounding graph (same rules as EntityGroundingGraph) restricted to
    text-bearing elements. ``extra_nodes`` = [(node_id, attrs, [neighbour node ids])], used to
    attach e.g. a FIGURE_3D node carrying a 3D embedding.
    """
    from src.grounding.entity_graph import EntityGroundingGraph
    from src.utils.config import ConfigDict
    text_elems = [e for e in elements if e.get("type") in TEXT_ELEMENT_TYPES]
    builder = EntityGroundingGraph(ConfigDict({"ablation_switches": {"graph_enabled": False}}), checkpoint_path="")
    g = builder.build_graph(text_elems, entities, relations or [], run_gnn=False).copy()
    for node_id, attrs, nbrs in extra_nodes or []:
        g.add_node(node_id, **attrs)
        for n in nbrs:
            if g.has_node(n):
                g.add_edge(node_id, n, relation="OTHER")
    return g


def candidate_pairs(elements: List[Dict[str, Any]], entities: List[Dict[str, Any]]
                    ) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    return [(el, en) for el in elements if el.get("type") in TEXT_ELEMENT_TYPES for en in entities]
