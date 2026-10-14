"""
Controlled Scientific Figure Benchmark Generator ("SciFig-Controlled").

Renders scientific figures with matplotlib so that every figure has exact ground truth:
figure elements (text + pixel bbox), scientific entities, gold element -> entity grounding links,
whether the figure is 3D, and (for 3D figures) the underlying 3D geometry.

Honesty notes
-------------
* This is a SYNTHETIC, controlled benchmark. Figures, paragraphs and captions are generated
  from templates. Results on it measure behaviour under controlled conditions and do not
  transfer automatically to real papers.
* Molecular 3D figures are rendered from REAL experimental PDB structures (see
  data/raw/pdb/SOURCES.md). Surface plots and 3D point clouds use analytic geometry.
* Figures use display aliases ("GAT", "Top-1 Acc. (%)") while paragraphs use canonical names
  ("Graph Attention Network"), and paragraphs mention distractor entities that are not plotted,
  so grounding cannot be solved by substring matching alone.

Usage:
    python -m src.data.benchmark_generator --out data/benchmark --seed 42
"""

import argparse
import gzip
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ---------------------------------------------------------------------------
# Vocabulary: canonical entity name -> figure display aliases
# ---------------------------------------------------------------------------
METHODS: Dict[str, List[str]] = {
    "Graph Attention Network": ["GAT"],
    "Graph Convolutional Network": ["GCN"],
    "GraphSAGE": ["GraphSAGE", "SAGE"],
    "ResNet-50": ["ResNet-50", "RN-50"],
    "Vision Transformer": ["ViT-B/16", "ViT"],
    "PointNet++": ["PointNet++", "PN++"],
    "Dynamic Graph CNN": ["DGCNN"],
    "Long Short-Term Memory": ["LSTM"],
    "Random Forest": ["Random Forest", "RF"],
    "Gradient Boosting": ["XGBoost", "GBDT"],
    "U-Net": ["U-Net", "UNet"],
    "SchNet": ["SchNet"],
    "Equivariant Graph Neural Network": ["EGNN"],
    "Multilayer Perceptron": ["MLP"],
    "Support Vector Machine": ["SVM"],
    "BERT": ["BERT-base", "BERT"],
    "Logistic Regression": ["LogReg"],
    "Mask R-CNN": ["Mask R-CNN"],
    "DimeNet": ["DimeNet++", "DimeNet"],
    "Transformer": ["Transformer"],
}
DATASETS = ["ImageNet", "CIFAR-10", "QM9", "ModelNet40", "ShapeNet", "MD17", "Cora",
            "PubMed", "SQuAD", "COCO", "MNIST", "PDBbind", "ZINC"]
METRICS: Dict[str, List[str]] = {
    "Top-1 Accuracy": ["Top-1 Acc. (%)", "Top-1 Accuracy (%)"],
    "Mean Absolute Error": ["MAE (eV)", "Mean Absolute Error"],
    "F1 Score": ["F1", "F1 score"],
    "Root Mean Square Deviation": ["RMSD (nm)", "RMSD"],
    "Training Loss": ["Train loss", "Training loss"],
    "Mean Squared Error": ["MSE"],
    "AUROC": ["ROC-AUC", "AUROC"],
    "Perplexity": ["PPL", "Perplexity"],
    "Intersection over Union": ["mIoU (%)", "IoU"],
    "Potential Energy": ["Energy (kcal/mol)", "Potential energy"],
    "Precision": ["Precision"],
    "Recall": ["Recall"],
    "Throughput": ["Throughput (img/s)"],
    "Free Energy": ["Free energy (kJ/mol)", "dG (kJ/mol)"],
}
VARIABLES: Dict[str, List[str]] = {
    "Epoch": ["Epoch", "Epochs"],
    "Learning Rate": ["LR", "Learning rate"],
    "Training Set Size": ["# train samples", "Train size"],
    "Temperature": ["Temperature (K)", "T (K)"],
    "Simulation Time": ["Time (ns)", "t (ns)"],
    "Number of Layers": ["# layers", "Depth"],
    "Noise Level": ["Noise std", "Noise level"],
    "Batch Size": ["Batch size"],
    "Pressure": ["Pressure (GPa)", "P (GPa)"],
    "Sequence Length": ["Seq. length", "Sequence length"],
    "Dropout Rate": ["Dropout", "Dropout rate"],
    "Model Size": ["Params (M)", "# params (M)"],
    "Wavelength": ["Wavelength (nm)"],
}
COLOR_VARIABLES: Dict[str, List[str]] = {
    "Residue Index": ["Residue index", "Residue #"],
    "B-factor": ["B-factor", "B-factor (A^2)"],
}

# Real PDB structures (file, canonical molecule name, display aliases, PDB id)
MOLECULES: List[Tuple[str, str, List[str], str]] = [
    ("19hc.pdb.gz", "Nine-haem Cytochrome c", ["Cytochrome c (19HC)", "19HC"], "19HC"),
    ("1a28.pdb.gz", "Progesterone Receptor", ["Progesterone receptor", "1A28"], "1A28"),
    ("1grm_single.pdb", "Gramicidin A", ["Gramicidin A", "1GRM"], "1GRM"),
    ("1hvr.pdb", "HIV-1 Protease", ["HIV-1 protease (1HVR)", "1HVR"], "1HVR"),
    ("1osm.pdb.gz", "OmpK36 Porin", ["OmpK36", "1OSM"], "1OSM"),
    ("4E43.pdb", "HIV Protease Variant", ["HIV protease (4E43)", "4E43"], "4E43"),
    ("5a7u.pdb", "ADR1 Zinc Finger", ["ADR1 zinc finger", "5A7U"], "5A7U"),
    ("adk_closed.pdb", "Adenylate Kinase Closed", ["ADK closed (1AKE)", "1AKE"], "1AKE"),
    ("adk_open.pdb", "Adenylate Kinase Open", ["ADK open (4AKE)", "4AKE"], "4AKE"),
    ("cobrotoxin.pdb", "Cobrotoxin", ["Cobrotoxin", "1V6P"], "1V6P"),
    ("nmr_neopetrosiamide.pdb", "Neopetrosiamide A", ["Neopetrosiamide", "2JUY"], "2JUY"),
    ("fullerene.pdb.gz", "C60 Fullerene", ["C60 fullerene", "C60"], "C60"),
    ("1A8O.pdb", "HIV Capsid Domain", ["HIV capsid CTD", "1A8O"], "1A8O"),
    ("1LCD.pdb", "Lac Repressor Headpiece", ["Lac repressor", "1LCD"], "1LCD"),
    ("2BEG.pdb", "Amyloid-beta Fibril", ["Abeta(1-42) fibril", "2BEG"], "2BEG"),
    ("2XHE.pdb", "Unc18 Syntaxin Complex", ["Unc18-syntaxin", "2XHE"], "2XHE"),
]
# Structure-disjoint test molecules (ADK open/closed kept together)
TEST_MOLECULES = {"1HVR", "1AKE", "4AKE", "2BEG", "5A7U"}

SURFACE_FAMILIES = ["gaussian", "saddle", "ripple", "paraboloid", "double_well"]
SHAPE_FAMILIES = ["sphere", "torus", "helix", "cube", "cylinder", "cone"]
FIG_TYPES_2D = ["line", "scatter", "bar", "heatmap"]
FIG_TYPES_3D = ["protein", "surface", "scatter3d"]
NUM_POINTS = 1024


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def load_pdb_coords(path: str) -> np.ndarray:
    """Cα coordinates of the first model (all chains); all heavy atoms if no Cα exist.
    Returns a list-of-chains concatenated in chain order, shape [N, 3]."""
    opener = gzip.open if path.endswith(".gz") else open
    ca, heavy = [], []
    with opener(path, "rt") as f:
        for line in f:
            if line.startswith("ENDMDL"):
                break
            if not line.startswith(("ATOM", "HETATM")):
                continue
            name = line[12:16].strip()
            resname = line[17:20].strip()
            if resname in ("HOH", "WAT", "TIP3", "SOL", "NA", "CL", "SOD", "CLA", "POT"):
                continue
            try:
                xyz = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
            except ValueError:
                continue
            if name == "CA" and line.startswith("ATOM"):
                ca.append(xyz)
            if not name.startswith("H"):
                heavy.append(xyz)
    pts = np.array(ca if len(ca) >= 10 else heavy, dtype=np.float32)
    if len(pts) == 0:
        raise ValueError(f"No coordinates parsed from {path}")
    return pts


def normalize_points(pts: np.ndarray) -> np.ndarray:
    pts = pts - pts.mean(0, keepdims=True)
    scale = np.linalg.norm(pts, axis=1).max()
    return (pts / max(scale, 1e-8)).astype(np.float32)


def resample_chain(pts: np.ndarray, n: int = NUM_POINTS) -> np.ndarray:
    """Resample an ordered trace to exactly n points by linear interpolation along its index."""
    if len(pts) == n:
        return pts.astype(np.float32)
    src = np.linspace(0, 1, len(pts))
    dst = np.linspace(0, 1, n)
    return np.stack([np.interp(dst, src, pts[:, d]) for d in range(3)], 1).astype(np.float32)


def surface_geometry(family: str, rng: np.random.Generator, grid: int = 32):
    a, b = rng.uniform(0.6, 1.6, 2)
    x, y = np.meshgrid(np.linspace(-2, 2, grid), np.linspace(-2, 2, grid))
    if family == "gaussian":
        z = np.exp(-(a * x ** 2 + b * y ** 2))
    elif family == "saddle":
        z = a * x ** 2 - b * y ** 2
    elif family == "ripple":
        r = np.sqrt(x ** 2 + y ** 2) + 1e-6
        z = np.sin(3 * a * r) / (r + b)
    elif family == "paraboloid":
        z = a * x ** 2 + b * y ** 2
    else:  # double_well
        z = (x ** 2 - a) ** 2 + b * y ** 2
    return x, y, z


def shape_points(family: str, rng: np.random.Generator, n: int = 300) -> np.ndarray:
    u = rng.uniform(0, 2 * np.pi, n)
    v = rng.uniform(0, 2 * np.pi, n)
    if family == "sphere":
        phi = np.arccos(rng.uniform(-1, 1, n))
        p = np.stack([np.sin(phi) * np.cos(u), np.sin(phi) * np.sin(u), np.cos(phi)], 1)
    elif family == "torus":
        R, r = rng.uniform(1.0, 1.5), rng.uniform(0.25, 0.5)
        p = np.stack([(R + r * np.cos(v)) * np.cos(u), (R + r * np.cos(v)) * np.sin(u), r * np.sin(v)], 1)
    elif family == "helix":
        t = np.sort(rng.uniform(0, rng.uniform(4, 8) * np.pi, n))
        p = np.stack([np.cos(t), np.sin(t), t / t.max() * 2 - 1], 1)
    elif family == "cube":
        p = rng.uniform(-1, 1, (n, 3))
        ax = rng.integers(0, 3, n)
        p[np.arange(n), ax] = np.sign(rng.uniform(-1, 1, n))
    elif family == "cylinder":
        h = rng.uniform(-1, 1, n)
        p = np.stack([np.cos(u), np.sin(u), h * rng.uniform(1, 1.8)], 1)
    else:  # cone
        h = rng.uniform(0, 1, n)
        p = np.stack([h * np.cos(u), h * np.sin(u), 1 - 2 * h], 1)
    p = p + rng.normal(0, 0.02, p.shape)
    return p.astype(np.float32)


# ---------------------------------------------------------------------------
# Figure element bookkeeping
# ---------------------------------------------------------------------------
def _bbox_px(artist, fig, renderer) -> List[int]:
    bb = artist.get_window_extent(renderer=renderer)
    W, H = fig.get_size_inches() * fig.dpi
    clip = lambda v, hi: int(min(max(v, 0), hi))
    return [clip(bb.x0, W), clip(H - bb.y1, H), clip(bb.x1, W), clip(H - bb.y0, H)]


class _FigRecord:
    def __init__(self):
        self.elements: List[Dict[str, Any]] = []
        self._artists: List[Any] = []

    def add(self, etype: str, text: str, artist, link: Optional[Tuple[str, str]] = None):
        """link = (canonical entity text, relation) or None for an unlinked (negative) element."""
        eid = f"{etype.lower()}_{len(self.elements)}"
        self.elements.append({"id": eid, "type": etype, "text": text,
                              "gold_entity": link[0] if link else None,
                              "gold_relation": link[1] if link else None})
        self._artists.append(artist)

    def finalize(self, fig):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for el, art in zip(self.elements, self._artists):
            el["bbox"] = _bbox_px(art, fig, renderer)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------
class BenchmarkGenerator:
    def __init__(self, out_dir: str, pdb_dir: str = "data/raw/pdb", seed: int = 42,
                 n_per_2d_type: int = 50, n_views_per_molecule: int = 5,
                 n_surface: int = 60, n_scatter3d: int = 60, test_fraction: float = 0.3):
        self.out_dir = out_dir
        self.pdb_dir = pdb_dir
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.n_per_2d_type = n_per_2d_type
        self.n_views = n_views_per_molecule
        self.n_surface = n_surface
        self.n_scatter3d = n_scatter3d
        self.test_fraction = test_fraction
        os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "geometry"), exist_ok=True)

    # ---- text helpers ---------------------------------------------------
    def _pick(self, d, k):
        keys = list(d.keys()) if isinstance(d, dict) else list(d)
        idx = self.rng.choice(len(keys), size=k, replace=False)
        return [keys[i] for i in idx]

    def _alias(self, table: Dict[str, List[str]], canon: str) -> str:
        opts = table[canon]
        return opts[int(self.rng.integers(len(opts)))]

    def _mention(self, canon: str, alias: str) -> str:
        """Canonical name, sometimes with the figure abbreviation defined in parentheses."""
        if alias.lower() != canon.lower() and self.rng.random() < 0.5:
            short = alias.split(" (")[0]
            return f"{canon} ({short})"
        return canon

    @staticmethod
    def _ent(text: str, etype: str) -> Dict[str, str]:
        return {"text": text, "type": etype}

    # ---- 2D figures -----------------------------------------------------
    def _fig_xy(self, kind: str, rec: _FigRecord):
        n_m = int(self.rng.integers(2, 5))
        methods = self._pick(METHODS, n_m)
        metric = self._pick(METRICS, 1)[0]
        var = self._pick(VARIABLES, 1)[0]
        dataset = self._pick(DATASETS, 1)[0]
        m_alias = {m: self._alias(METHODS, m) for m in methods}
        y_alias, x_alias = self._alias(METRICS, metric), self._alias(VARIABLES, var)

        fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=100)
        x = np.linspace(0, 1, 12)
        for m in methods:
            y = np.cumsum(self.rng.normal(0.1, 0.05, len(x)))
            if kind == "line":
                ax.plot(x, y, label=m_alias[m], linewidth=2)
            else:
                ax.scatter(x + self.rng.normal(0, 0.01, len(x)), y, label=m_alias[m], s=25)
        xl = ax.set_xlabel(x_alias, fontsize=13)
        yl = ax.set_ylabel(y_alias, fontsize=13)
        leg = ax.legend(fontsize=12, loc="upper left")
        rec.add("AXIS", x_alias, xl, (var, "REPRESENTS"))
        rec.add("AXIS", y_alias, yl, (metric, "REPRESENTS"))
        for m, t in zip(methods, leg.get_texts()):
            rec.add("LEGEND", m_alias[m], t, (m, "CORRESPONDS_TO"))
        ents = [self._ent(m, "METHOD") for m in methods] + [
            self._ent(metric, "METRIC"), self._ent(var, "VARIABLE"), self._ent(dataset, "DATASET")]
        mentions = [self._mention(m, m_alias[m]) for m in methods]
        para = (f"We compare {', '.join(mentions)} on {dataset}. "
                f"The figure reports {self._mention(metric, y_alias)} as a function of "
                f"{self._mention(var, x_alias)}.")
        caption = (f"{metric} versus {var} for {', '.join(methods)} on {dataset}.")
        return fig, ents, para, caption

    def _fig_bar(self, rec: _FigRecord):
        n_m = int(self.rng.integers(3, 6))
        methods = self._pick(METHODS, n_m)
        metric = self._pick(METRICS, 1)[0]
        dataset = self._pick(DATASETS, 1)[0]
        m_alias = {m: self._alias(METHODS, m) for m in methods}
        y_alias = self._alias(METRICS, metric)
        fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=100)
        ax.bar(range(n_m), self.rng.uniform(0.3, 1.0, n_m), color="C0")
        ax.set_xticks(range(n_m))
        ticks = ax.set_xticklabels([m_alias[m] for m in methods], fontsize=12, rotation=0)
        yl = ax.set_ylabel(y_alias, fontsize=13)
        xl = ax.set_xlabel("Method", fontsize=13)
        rec.add("AXIS", "Method", xl, None)
        rec.add("AXIS", y_alias, yl, (metric, "REPRESENTS"))
        for m, t in zip(methods, ticks):
            rec.add("LABEL", m_alias[m], t, (m, "CORRESPONDS_TO"))
        ents = [self._ent(m, "METHOD") for m in methods] + [
            self._ent(metric, "METRIC"), self._ent(dataset, "DATASET")]
        para = (f"Results of {', '.join(self._mention(m, m_alias[m]) for m in methods)} on "
                f"{dataset} are summarised by {self._mention(metric, y_alias)}.")
        caption = f"{metric} of {', '.join(methods)} on {dataset}."
        return fig, ents, para, caption

    def _fig_heatmap(self, rec: _FigRecord):
        v1, v2 = self._pick(VARIABLES, 2)
        metric = self._pick(METRICS, 1)[0]
        method = self._pick(METHODS, 1)[0]
        dataset = self._pick(DATASETS, 1)[0]
        a1, a2 = self._alias(VARIABLES, v1), self._alias(VARIABLES, v2)
        c_alias = self._alias(METRICS, metric)
        fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=100)
        im = ax.imshow(self.rng.random((8, 8)), cmap=str(self.rng.choice(["viridis", "magma", "Blues"])))
        cb = fig.colorbar(im, ax=ax)
        cb.set_label(c_alias, fontsize=13)
        cl = cb.ax.yaxis.label
        xl = ax.set_xlabel(a1, fontsize=13)
        yl = ax.set_ylabel(a2, fontsize=13)
        rec.add("AXIS", a1, xl, (v1, "REPRESENTS"))
        rec.add("AXIS", a2, yl, (v2, "REPRESENTS"))
        rec.add("LABEL", c_alias, cl, (metric, "REPRESENTS"))
        ents = [self._ent(method, "METHOD"), self._ent(metric, "METRIC"), self._ent(v1, "VARIABLE"),
                self._ent(v2, "VARIABLE"), self._ent(dataset, "DATASET")]
        para = (f"For {method} on {dataset}, we sweep {self._mention(v1, a1)} and "
                f"{self._mention(v2, a2)} and report {self._mention(metric, c_alias)}.")
        caption = f"{metric} of {method} over {v1} and {v2} on {dataset}."
        return fig, ents, para, caption

    # ---- 3D figures -----------------------------------------------------
    def _new_3d(self):
        fig = plt.figure(figsize=(6.4, 4.8), dpi=100)
        ax = fig.add_subplot(111, projection="3d")
        ax.view_init(elev=float(self.rng.uniform(10, 50)), azim=float(self.rng.uniform(0, 360)))
        return fig, ax

    def _fig_protein(self, rec: _FigRecord, mol):
        fname, canon, aliases, pdb_id = mol
        pts = load_pdb_coords(os.path.join(self.pdb_dir, fname))
        cvar = self._pick(COLOR_VARIABLES, 1)[0]
        c_alias = self._alias(COLOR_VARIABLES, cvar)
        title_alias = aliases[int(self.rng.integers(len(aliases)))]
        fig, ax = self._new_3d()
        colors = np.arange(len(pts)) if cvar == "Residue Index" else self.rng.random(len(pts))
        if len(pts) < 2000:
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color="0.6", linewidth=0.8)
        sc = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=colors,
                        cmap=str(self.rng.choice(["viridis", "plasma", "coolwarm"])), s=8)
        cb = fig.colorbar(sc, ax=ax, shrink=0.7)
        cb.set_label(c_alias, fontsize=13)
        title = ax.set_title(title_alias, fontsize=14)
        xl = ax.set_xlabel("x (A)", fontsize=12)
        yl = ax.set_ylabel("y (A)", fontsize=12)
        zl = ax.set_zlabel("z (A)", fontsize=12)
        rec.add("LABEL", title_alias, title, (canon, "REPRESENTS"))
        rec.add("LABEL", c_alias, cb.ax.yaxis.label, (cvar, "REPRESENTS"))
        for t, art in (("x (A)", xl), ("y (A)", yl), ("z (A)", zl)):
            rec.add("AXIS", t, art, None)
        distract = self._pick(METHODS, 1)[0]
        ents = [self._ent(canon, "MATERIAL"), self._ent(cvar, "VARIABLE"),
                self._ent("Protein Data Bank", "DATASET"), self._ent(distract, "METHOD")]
        para = (f"The experimental structure of {canon} (PDB {pdb_id}) is shown as a C-alpha trace "
                f"coloured by {cvar}. Structures were taken from the Protein Data Bank; "
                f"{distract} is used elsewhere in this work.")
        caption = f"3D C-alpha trace of {canon} (PDB {pdb_id}) coloured by {cvar}."
        geom = normalize_points(resample_chain(pts))
        return fig, ents, para, caption, geom, {"pdb_id": pdb_id, "pdb_file": fname}

    def _fig_surface(self, rec: _FigRecord):
        family = SURFACE_FAMILIES[int(self.rng.integers(len(SURFACE_FAMILIES)))]
        v1, v2 = self._pick(VARIABLES, 2)
        metric = self._pick(METRICS, 1)[0]
        method = self._pick(METHODS, 1)[0]
        a1, a2, zm = self._alias(VARIABLES, v1), self._alias(VARIABLES, v2), self._alias(METRICS, metric)
        x, y, z = surface_geometry(family, self.rng)
        fig, ax = self._new_3d()
        ax.plot_surface(x, y, z, cmap=str(self.rng.choice(["viridis", "coolwarm", "terrain"])),
                        linewidth=0, antialiased=True)
        xl = ax.set_xlabel(a1, fontsize=12)
        yl = ax.set_ylabel(a2, fontsize=12)
        zl = ax.set_zlabel(zm, fontsize=12)
        rec.add("AXIS", a1, xl, (v1, "REPRESENTS"))
        rec.add("AXIS", a2, yl, (v2, "REPRESENTS"))
        rec.add("AXIS", zm, zl, (metric, "REPRESENTS"))
        ents = [self._ent(method, "METHOD"), self._ent(metric, "METRIC"),
                self._ent(v1, "VARIABLE"), self._ent(v2, "VARIABLE")]
        para = (f"The {self._mention(metric, zm)} landscape of {method} is shown as a surface over "
                f"{self._mention(v1, a1)} and {self._mention(v2, a2)}.")
        caption = f"3D surface of {metric} for {method} over {v1} and {v2}."
        geom = np.stack([x.ravel(), y.ravel(), (z / max(np.abs(z).max(), 1e-8)).ravel() * 2], 1)
        geom = normalize_points(geom)
        return fig, ents, para, caption, geom, {"surface_family": family}

    def _fig_scatter3d(self, rec: _FigRecord):
        n_m = int(self.rng.integers(2, 4))
        methods = self._pick(METHODS, n_m)
        dataset = self._pick(DATASETS, 1)[0]
        m_alias = {m: self._alias(METHODS, m) for m in methods}
        fams = [SHAPE_FAMILIES[int(i)] for i in self.rng.integers(0, len(SHAPE_FAMILIES), n_m)]
        fig, ax = self._new_3d()
        clouds = []
        for i, (m, fam) in enumerate(zip(methods, fams)):
            p = shape_points(fam, self.rng) * 0.6 + np.array([i * 1.6 - (n_m - 1) * 0.8, 0, 0])
            clouds.append(p)
            ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=6, label=m_alias[m])
        leg = ax.legend(fontsize=12, loc="upper left")
        xl = ax.set_xlabel("PC1", fontsize=12)
        yl = ax.set_ylabel("PC2", fontsize=12)
        zl = ax.set_zlabel("PC3", fontsize=12)
        for t, art in (("PC1", xl), ("PC2", yl), ("PC3", zl)):
            rec.add("AXIS", t, art, None)
        for m, t in zip(methods, leg.get_texts()):
            rec.add("LEGEND", m_alias[m], t, (m, "CORRESPONDS_TO"))
        distract = [d for d in self._pick(METHODS, n_m + 2) if d not in methods][:2]
        ents = [self._ent(m, "METHOD") for m in methods + distract] + [self._ent(dataset, "DATASET")]
        para = (f"Learned 3D feature embeddings of {', '.join(self._mention(m, m_alias[m]) for m in methods)} "
                f"on {dataset}, projected with PCA. We also discuss {', '.join(distract)}.")
        caption = f"3D PCA embedding of features from {', '.join(methods)} on {dataset}."
        allp = np.concatenate(clouds)
        idx = self.rng.choice(len(allp), NUM_POINTS, replace=len(allp) < NUM_POINTS)
        geom = normalize_points(allp[idx])
        return fig, ents, para, caption, geom, {"shape_families": fams}

    # ---- driver ---------------------------------------------------------
    def _add_distractors(self, ents: List[Dict[str, str]], para: str) -> Tuple[List[Dict[str, str]], str]:
        """Mention 1-2 methods and one metric in the paragraph that are NOT in the figure."""
        have = {e["text"] for e in ents}
        extra_m = [m for m in self._pick(METHODS, 4) if m not in have][: int(self.rng.integers(1, 3))]
        extra_k = [k for k in self._pick(METRICS, 3) if k not in have][:1]
        if extra_m or extra_k:
            para += (f" Prior work used {', '.join(extra_m + extra_k)}"
                     f"{' as reported in the appendix' if self.rng.random() < 0.5 else ''}.")
        ents = ents + [self._ent(m, "METHOD") for m in extra_m] + [self._ent(k, "METRIC") for k in extra_k]
        return ents, para

    def _save(self, fig_id, fig, rec, kind, is_3d, ents, para, caption, split, geom=None, extra=None):
        rec.finalize(fig)
        img_rel = f"images/{fig_id}.png"
        fig.savefig(os.path.join(self.out_dir, img_rel), dpi=100)
        plt.close(fig)
        geom_rel = None
        if geom is not None:
            geom_rel = f"geometry/{fig_id}.npy"
            np.save(os.path.join(self.out_dir, geom_rel), geom.astype(np.float32))
        # de-duplicate entities preserving order
        seen, uniq = set(), []
        for e in ents:
            if e["text"] not in seen:
                seen.add(e["text"])
                uniq.append(e)
        return {
            "figure_id": fig_id, "split": split, "figure_type": kind, "is_3d": is_3d,
            "image_path": img_rel, "geometry_path": geom_rel,
            "paragraph_text": para, "caption": caption,
            "entities": uniq, "elements": rec.elements,
            "gold_links": [{"element_id": el["id"], "entity": el["gold_entity"], "relation": el["gold_relation"]}
                           for el in rec.elements if el["gold_entity"]],
            **(extra or {}),
        }

    def _split(self) -> str:
        return "test" if self.rng.random() < self.test_fraction else "train"

    def generate(self) -> Dict[str, Any]:
        figures: List[Dict[str, Any]] = []
        k = 0
        for kind in FIG_TYPES_2D:
            for _ in range(self.n_per_2d_type):
                rec = _FigRecord()
                if kind in ("line", "scatter"):
                    fig, ents, para, cap = self._fig_xy(kind, rec)
                elif kind == "bar":
                    fig, ents, para, cap = self._fig_bar(rec)
                else:
                    fig, ents, para, cap = self._fig_heatmap(rec)
                ents, para = self._add_distractors(ents, para)
                figures.append(self._save(f"fig_{k:04d}", fig, rec, kind, False, ents, para, cap, self._split()))
                k += 1
        for mol in MOLECULES:
            split = "test" if mol[3] in TEST_MOLECULES else "train"
            for _ in range(self.n_views):
                rec = _FigRecord()
                fig, ents, para, cap, geom, extra = self._fig_protein(rec, mol)
                figures.append(self._save(f"fig_{k:04d}", fig, rec, "protein", True, ents, para, cap, split, geom, extra))
                k += 1
        for _ in range(self.n_surface):
            rec = _FigRecord()
            fig, ents, para, cap, geom, extra = self._fig_surface(rec)
            ents, para = self._add_distractors(ents, para)
            figures.append(self._save(f"fig_{k:04d}", fig, rec, "surface", True, ents, para, cap, self._split(), geom, extra))
            k += 1
        for _ in range(self.n_scatter3d):
            rec = _FigRecord()
            fig, ents, para, cap, geom, extra = self._fig_scatter3d(rec)
            figures.append(self._save(f"fig_{k:04d}", fig, rec, "scatter3d", True, ents, para, cap, self._split(), geom, extra))
            k += 1

        meta = {
            "dataset_name": "SciFig-Controlled",
            "version": "1.0",
            "description": ("Synthetic, matplotlib-rendered scientific figures with exact ground truth. "
                            "Molecular 3D figures use real PDB structures (data/raw/pdb/SOURCES.md)."),
            "seed": self.seed,
            "matplotlib_version": matplotlib.__version__,
            "num_figures": len(figures),
            "num_train": sum(f["split"] == "train" for f in figures),
            "num_test": sum(f["split"] == "test" for f in figures),
            "test_molecules": sorted(TEST_MOLECULES),
            "figures": figures,
        }
        with open(os.path.join(self.out_dir, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=1)
        return meta


def load_benchmark(path: str = "data/benchmark") -> Dict[str, Any]:
    with open(os.path.join(path, "metadata.json")) as f:
        meta = json.load(f)
    meta["root"] = path
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/benchmark")
    ap.add_argument("--pdb-dir", default="data/raw/pdb")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-per-2d-type", type=int, default=50)
    ap.add_argument("--n-views", type=int, default=5)
    ap.add_argument("--n-surface", type=int, default=60)
    ap.add_argument("--n-scatter3d", type=int, default=60)
    a = ap.parse_args()
    meta = BenchmarkGenerator(a.out, a.pdb_dir, a.seed, a.n_per_2d_type, a.n_views, a.n_surface, a.n_scatter3d).generate()
    print(f"Wrote {meta['num_figures']} figures ({meta['num_train']} train / {meta['num_test']} test) to {a.out}")


if __name__ == "__main__":
    main()
