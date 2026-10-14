"""
PyTorch Geometric Graph Converter Module
Converts NetworkX Entity Grounding Graphs into PyTorch Geometric Data / HeteroData structures.
Preserves entity types, edge relations, layout attributes, and constructs initial node feature vectors.
"""

from typing import Dict, Any, List, Tuple
import torch
import numpy as np
import networkx as nx
from torch_geometric.data import Data, HeteroData

# Global mapping dictionaries for categorical feature encoding
NODE_TYPES = ["FIGURE_ELEMENT", "SCIENTIFIC_ENTITY", "PAPER_CLAIM", "PARAGRAPH_CONTEXT", "FIGURE_3D", "FIGURE_IMAGE"]
ELEMENT_TYPES = ["AXIS", "LEGEND", "CHART_REGION", "LABEL", "NONE"]
ENTITY_TYPES = [
    "METHOD", "MODEL", "DATASET", "METRIC", "METRIC_VALUE", 
    "TASK", "MATERIAL", "ALGORITHM", "CLAIM", "VARIABLE", "ORGANIZATION", "NONE"
]

RELATION_TYPES = [
    "REPRESENTS", "CORRESPONDS_TO", "ACHIEVES", "EVALUATED_ON", 
    "COMPARED_WITH", "SUPPORTED_BY", "MENTIONED_IN", "OTHER"
]

class PyGGraphConverter:
    def __init__(self, feature_dim: int = 64, text_encoding: str = "positional"):
        """
        text_encoding:
            "positional" - original character x position hashing (default, backward compatible)
            "trigram"    - bag of hashed character trigrams (order-free, robust to small edits)
        Nodes carrying a ``modal_embedding`` attribute (e.g. a FIGURE_3D node with a 3D embedding)
        use that vector, truncated/zero-padded and L2-normalised, in place of the text features.
        """
        self.feature_dim = feature_dim
        self.text_encoding = text_encoding

    def convert_to_pyg_data(self, graph: nx.DiGraph) -> Tuple[Data, Dict[str, int], Dict[int, str]]:
        """
        Converts a NetworkX DiGraph into a PyTorch Geometric Data object.
        Returns:
            - pyg_data: torch_geometric.data.Data containing x (node features), edge_index, edge_attr
            - node2idx: mapping from string node ID to integer index
            - idx2node: mapping from integer index to string node ID
        """
        nodes = list(graph.nodes(data=True))
        node2idx = {node_id: i for i, (node_id, _) in enumerate(nodes)}
        idx2node = {i: node_id for i, (node_id, _) in enumerate(nodes)}
        
        num_nodes = len(nodes)
        if num_nodes == 0:
            # Empty graph fallback
            dummy_x = torch.zeros((1, self.feature_dim), dtype=torch.float32)
            dummy_edge_index = torch.empty((2, 0), dtype=torch.long)
            return Data(x=dummy_x, edge_index=dummy_edge_index), {}, {}

        # 1. Construct Node Feature Matrix X
        node_features = []
        for node_id, data in nodes:
            feat_vec = self._encode_node_features(data)
            node_features.append(feat_vec)

        x_tensor = torch.tensor(np.array(node_features), dtype=torch.float32)

        # 2. Construct Edge Index & Edge Attributes
        edge_index_list = []
        edge_attr_list = []

        for u, v, data in graph.edges(data=True):
            if u in node2idx and v in node2idx:
                src_idx = node2idx[u]
                tgt_idx = node2idx[v]
                edge_index_list.append([src_idx, tgt_idx])
                
                rel = data.get("relation", "OTHER")
                rel_idx = RELATION_TYPES.index(rel) if rel in RELATION_TYPES else RELATION_TYPES.index("OTHER")
                
                # One-hot edge relation attribute
                rel_onehot = [0.0] * len(RELATION_TYPES)
                rel_onehot[rel_idx] = 1.0
                edge_attr_list.append(rel_onehot)

        if len(edge_index_list) > 0:
            edge_index_tensor = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
            edge_attr_tensor = torch.tensor(np.array(edge_attr_list), dtype=torch.float32)
        else:
            edge_index_tensor = torch.empty((2, 0), dtype=torch.long)
            edge_attr_tensor = torch.empty((0, len(RELATION_TYPES)), dtype=torch.float32)

        pyg_data = Data(
            x=x_tensor,
            edge_index=edge_index_tensor,
            edge_attr=edge_attr_tensor,
            num_nodes=num_nodes
        )

        return pyg_data, node2idx, idx2node

    def convert_to_hetero_data(self, graph: nx.DiGraph) -> Tuple[HeteroData, Dict[str, Tuple[str, int]]]:
        """
        Converts a NetworkX DiGraph into a PyTorch Geometric HeteroData structure.
        """
        hetero_data = HeteroData()
        node_type_idx_map = {}

        # Group nodes by type
        typed_nodes: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
        for node_id, data in graph.nodes(data=True):
            ntype = data.get("node_type", "SCIENTIFIC_ENTITY")
            if ntype not in typed_nodes:
                typed_nodes[ntype] = []
            typed_nodes[ntype].append((node_id, data))

        # Build feature tensors per node type
        for ntype, nlist in typed_nodes.items():
            feats = [self._encode_node_features(d) for _, d in nlist]
            hetero_data[ntype].x = torch.tensor(np.array(feats), dtype=torch.float32)
            for idx, (node_id, _) in enumerate(nlist):
                node_type_idx_map[node_id] = (ntype, idx)

        # Build edge indices per relation tuple (src_type, relation, tgt_type)
        hetero_edges: Dict[Tuple[str, str, str], List[Tuple[int, int]]] = {}
        for u, v, data in graph.edges(data=True):
            if u in node_type_idx_map and v in node_type_idx_map:
                src_type, src_idx = node_type_idx_map[u]
                tgt_type, tgt_idx = node_type_idx_map[v]
                rel = data.get("relation", "REL")
                edge_tuple = (src_type, rel, tgt_type)
                
                if edge_tuple not in hetero_edges:
                    hetero_edges[edge_tuple] = []
                hetero_edges[edge_tuple].append((src_idx, tgt_idx))

        for edge_tuple, elist in hetero_edges.items():
            e_tensor = torch.tensor(elist, dtype=torch.long).t().contiguous()
            hetero_data[edge_tuple].edge_index = e_tensor

        return hetero_data, node_type_idx_map

    def _encode_node_features(self, data: Dict[str, Any]) -> np.ndarray:
        """
        Encodes node attributes (node_type, element_type, entity_type, label hash) into a dense feature vector.
        """
        # 1. One-hot node type (len NODE_TYPES)
        ntype = data.get("node_type", "SCIENTIFIC_ENTITY")
        ntype_onehot = [0.0] * len(NODE_TYPES)
        if ntype in NODE_TYPES:
            ntype_onehot[NODE_TYPES.index(ntype)] = 1.0

        # 2. One-hot element type (len ELEMENT_TYPES)
        elem_type = data.get("element_type", "NONE")
        elem_onehot = [0.0] * len(ELEMENT_TYPES)
        if elem_type in ELEMENT_TYPES:
            elem_onehot[ELEMENT_TYPES.index(elem_type)] = 1.0

        # 3. One-hot entity type (len ENTITY_TYPES)
        ent_type = data.get("entity_type", "NONE")
        ent_onehot = [0.0] * len(ENTITY_TYPES)
        if ent_type in ENTITY_TYPES:
            ent_onehot[ENTITY_TYPES.index(ent_type)] = 1.0

        cat_vec = np.array(ntype_onehot + elem_onehot + ent_onehot, dtype=np.float32)

        # 4. Text label feature projection via hash embedding
        target_dim = self.feature_dim - len(cat_vec)
        modal = data.get("modal_embedding")
        if modal is not None:
            text_feat = np.zeros(target_dim, dtype=np.float32)
            m = np.asarray(modal, dtype=np.float32)[:target_dim]
            text_feat[:len(m)] = m
            n = np.linalg.norm(text_feat)
            if n > 0:
                text_feat = text_feat / n
        else:
            label_text = str(data.get("label", ""))
            text_feat = self._text_to_feature_vec(label_text, target_dim=target_dim)

        full_vec = np.concatenate([cat_vec, text_feat])
        return full_vec

    def _text_to_feature_vec(self, text: str, target_dim: int) -> np.ndarray:
        """
        Deterministic word-level character n-gram hashing vector for text labels.
        """
        vec = np.zeros(target_dim, dtype=np.float32)
        if not text:
            return vec
        text_clean = text.lower()
        if self.text_encoding == "trigram":
            import zlib
            padded = f"  {text_clean} "
            for i in range(len(padded) - 2):
                vec[zlib.crc32(padded[i:i + 3].encode()) % target_dim] += 1.0
        else:
            for i, char in enumerate(text_clean):
                idx = (ord(char) * (i + 1)) % target_dim
                vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec
