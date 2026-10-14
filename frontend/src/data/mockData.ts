import type { PipelineResult, DocumentStructure } from '../types';

export const SAMPLE_DOCUMENT: DocumentStructure = {
  paper_id: "arxiv_2103_ResNet_ViT_01",
  title: "Deep Residual Learning vs. Vision Transformers for Large-Scale Visual Recognition",
  num_pages: 12,
  sections: [
    {
      section_id: "sec_1",
      title: "1. Abstract & Introduction",
      paragraphs: [
        {
          paragraph_id: "p_1",
          page: 1,
          text: "Visual recognition architectures have evolved rapidly from convolutional networks like ResNet to attention-driven Vision Transformers (ViT). In this work, we evaluate ResNet-50 and ViT-Base on CIFAR-10 and ImageNet-1k datasets across 300 epochs.",
          figure_references: ["Figure 1", "Figure 3(a)"]
        },
        {
          paragraph_id: "p_2",
          page: 2,
          text: "ResNet-50 achieves 85.3% top-1 accuracy on CIFAR-10 under standard data augmentation, whereas ViT-Base achieves 88.1% top-1 accuracy when pretrained on JFT-300M.",
          figure_references: ["Figure 2"]
        }
      ]
    },
    {
      section_id: "sec_2",
      title: "2. Empirical Setup & Methodology",
      paragraphs: [
        {
          paragraph_id: "p_3",
          page: 4,
          text: "Both models were trained using AdamW optimizer with a base learning rate of 1e-3 and cosine learning rate decay over 300 epochs on Apple Silicon M4 Pro MPS acceleration.",
          figure_references: ["Figure 3"]
        }
      ]
    },
    {
      section_id: "sec_3",
      title: "3. Results & Convergence Analysis",
      paragraphs: [
        {
          paragraph_id: "p_4",
          page: 7,
          text: "Figure 3 demonstrates the top-1 accuracy curves across epochs. ResNet converges faster in early epochs (0-50), but ViT surpasses ResNet beyond epoch 120, demonstrating superior long-range dependency modeling.",
          figure_references: ["Figure 3", "Figure 3(a)"]
        }
      ]
    }
  ]
};

export const DEFAULT_PIPELINE_RESULT: PipelineResult = {
  is_mock_data: true,
  paper_id: "arxiv_2103_ResNet_ViT_01",
  figure_id: "fig_03_accuracy_curves",
  query: "What figure shows the accuracy comparison between ResNet and ViT over training epochs?",
  extracted_entities: [
    { text: "ResNet-50", type: "METHOD", confidence: 0.95 },
    { text: "ViT-Base", type: "MODEL", confidence: 0.94 },
    { text: "CIFAR-10", type: "DATASET", confidence: 0.98 },
    { text: "ImageNet-1k", type: "DATASET", confidence: 0.96 },
    { text: "Top-1 Accuracy", type: "METRIC", confidence: 0.97 },
    { text: "85.3%", type: "METRIC_VALUE", confidence: 0.99 },
    { text: "88.1%", type: "METRIC_VALUE", confidence: 0.99 },
    { text: "Epochs", type: "VARIABLE", confidence: 0.92 },
    { text: "Cosine Decay", type: "ALGORITHM", confidence: 0.89 },
    { text: "AdamW", type: "ALGORITHM", confidence: 0.91 }
  ],
  retrieved_context: {
    query: "What figure shows the accuracy comparison between ResNet and ViT over training epochs?",
    weighted_score: 0.8745,
    level_results: {
      paper: [
        {
          chunk_type: "paper",
          paper_id: "arxiv_2103_ResNet_ViT_01",
          text: "Deep Residual Learning vs. Vision Transformers for Large-Scale Visual Recognition. Evaluates ResNet-50 and ViT-Base on CIFAR-10 and ImageNet over 300 epochs.",
          score: 0.82
        }
      ],
      section: [
        {
          chunk_type: "section",
          paper_id: "arxiv_2103_ResNet_ViT_01",
          section_id: "sec_3",
          text: "3. Results & Convergence Analysis: Figure 3 demonstrates the top-1 accuracy curves across epochs. ResNet converges faster in early epochs (0-50), but ViT surpasses ResNet beyond epoch 120.",
          score: 0.89
        }
      ],
      paragraph: [
        {
          chunk_type: "paragraph",
          paper_id: "arxiv_2103_ResNet_ViT_01",
          section_id: "sec_3",
          paragraph_id: "p_4",
          text: "Figure 3 demonstrates the top-1 accuracy curves across epochs. ResNet converges faster in early epochs (0-50), but ViT surpasses ResNet beyond epoch 120, demonstrating superior long-range dependency modeling.",
          score: 0.94
        }
      ],
      entity: [
        {
          chunk_type: "entity",
          paper_id: "arxiv_2103_ResNet_ViT_01",
          entity_type: "METHOD",
          text: "ResNet-50 vs ViT-Base",
          score: 0.85
        }
      ]
    }
  },
  figure_elements: {
    figure_id: "fig_03_accuracy_curves",
    dimensions: { width: 640, height: 400 },
    status: "IMPLEMENTED (OpenCV Layout Extractor)",
    elements: [
      { id: "axis_x", type: "AXIS", text: "X-Axis: Training Epochs (0 - 300)", bbox: [60, 340, 560, 380] },
      { id: "axis_y", type: "AXIS", text: "Y-Axis: Top-1 Accuracy (%)", bbox: [10, 50, 50, 330] },
      { id: "legend_1", type: "LEGEND", text: "Legend Blue: ResNet-50 (Convolutional)", bbox: [420, 60, 610, 95] },
      { id: "legend_2", type: "LEGEND", text: "Legend Purple: ViT-Base (Self-Attention)", bbox: [420, 105, 610, 140] },
      { id: "chart_canvas", type: "CHART_REGION", text: "Accuracy Curve Canvas", bbox: [65, 50, 400, 330] }
    ]
  },
  grounding_graph_summary: {
    num_nodes: 9,
    num_edges: 7,
    nodes: [
      { id: "elem:axis_x", label: "Training Epochs", node_type: "FIGURE_ELEMENT", element_type: "AXIS" },
      { id: "elem:axis_y", label: "Top-1 Accuracy (%)", node_type: "FIGURE_ELEMENT", element_type: "AXIS" },
      { id: "elem:legend_1", label: "ResNet-50", node_type: "FIGURE_ELEMENT", element_type: "LEGEND" },
      { id: "elem:legend_2", label: "ViT-Base", node_type: "FIGURE_ELEMENT", element_type: "LEGEND" },
      { id: "ent:ResNet-50", label: "ResNet-50", node_type: "SCIENTIFIC_ENTITY", entity_type: "METHOD" },
      { id: "ent:ViT-Base", label: "ViT-Base", node_type: "SCIENTIFIC_ENTITY", entity_type: "MODEL" },
      { id: "ent:CIFAR-10", label: "CIFAR-10", node_type: "SCIENTIFIC_ENTITY", entity_type: "DATASET" },
      { id: "ent:Top-1 Accuracy", label: "Top-1 Accuracy", node_type: "SCIENTIFIC_ENTITY", entity_type: "METRIC" },
      { id: "ent:88.1%", label: "88.1% Accuracy", node_type: "SCIENTIFIC_ENTITY", entity_type: "METRIC_VALUE" }
    ],
    edges: [
      { source: "elem:axis_y", target: "ent:Top-1 Accuracy", relation: "REPRESENTS" },
      { source: "elem:legend_1", target: "ent:ResNet-50", relation: "CORRESPONDS_TO" },
      { source: "elem:legend_2", target: "ent:ViT-Base", relation: "CORRESPONDS_TO" },
      { source: "ent:ResNet-50", target: "ent:Top-1 Accuracy", relation: "ACHIEVES" },
      { source: "ent:ViT-Base", target: "ent:CIFAR-10", relation: "EVALUATED_ON" },
      { source: "ent:ViT-Base", target: "ent:88.1%", relation: "ACHIEVES" },
      { source: "ent:ResNet-50", target: "ent:ViT-Base", relation: "COMPARED_WITH" }
    ],
    status: "IMPLEMENTED (NetworkX Heterogeneous Grounding Graph)"
  },
  generated_caption: "Figure 3: Comparative top-1 accuracy curves for ResNet-50 and ViT-Base across 300 training epochs on CIFAR-10. ResNet-50 achieves early convergence (85.3% peak), while ViT-Base reaches 88.1% top-1 accuracy beyond epoch 120 under AdamW optimization.",
  verification_results: {
    technical_correctness_ratio: 1.0,
    status: "IMPLEMENTED (Rule & NLI Evidence Verification)",
    claims_verification: [
      {
        claim: "Comparative top-1 accuracy curves for ResNet-50 and ViT-Base across 300 training epochs on CIFAR-10",
        evidence_snippet: "ResNet-50 achieves 85.3% top-1 accuracy on CIFAR-10 under standard data augmentation...",
        status: "SUPPORTED"
      },
      {
        claim: "ResNet-50 achieves early convergence (85.3% peak)",
        evidence_snippet: "ResNet converges faster in early epochs (0-50)...",
        status: "SUPPORTED"
      },
      {
        claim: "ViT-Base reaches 88.1% top-1 accuracy beyond epoch 120 under AdamW optimization",
        evidence_snippet: "whereas ViT-Base achieves 88.1% top-1 accuracy when pretrained on JFT-300M...",
        status: "SUPPORTED"
      }
    ]
  },
  scq_evaluation: {
    scq_composite_score: 0.8875,
    breakdown: {
      entity_coverage: 0.90,
      technical_correctness: 1.00,
      reconstruction_fidelity: 0.85,
      retrieval_accuracy: 0.8745
    },
    weights_used: {
      w_entity_coverage: 0.30,
      w_technical_correctness: 0.30,
      w_reconstruction_fidelity: 0.20,
      w_retrieval_accuracy: 0.20
    }
  },
  status_declarations: {
    pdf_ingestion: "IMPLEMENTED (PyMuPDF structural parser)",
    scientific_ner: "IMPLEMENTED (SciBERT/Regex Hybrid)",
    hierarchical_retrieval: "IMPLEMENTED (4-Level Sentence Transformers + FAISS)",
    figure_processing: "IMPLEMENTED (OpenCV Layout Extractor)",
    entity_grounding_graph: "IMPLEMENTED (NetworkX Heterogeneous Grounding Graph)",
    grounded_captioning: "IMPLEMENTED (Grounded Template VLM Generator)",
    hallucination_verification: "IMPLEMENTED (Rule & NLI Evidence Verification)",
    cross_modal_tri_modal: "PARTIALLY IMPLEMENTED (CLIP Pretrained Embeddings)",
    "3d_visualization_pathway": "PLANNED (Phase 4)"
  }
};
