export type EntityType = 
  | 'METHOD' 
  | 'MODEL' 
  | 'DATASET' 
  | 'METRIC' 
  | 'METRIC_VALUE' 
  | 'TASK' 
  | 'MATERIAL' 
  | 'ALGORITHM' 
  | 'CLAIM' 
  | 'VARIABLE' 
  | 'ORGANIZATION' 
  | 'FIGURE_ELEMENT';

export interface Entity {
  text: string;
  type: EntityType;
  start?: number;
  end?: number;
  confidence: number;
}

export interface Relation {
  source: string;
  relation: string;
  target: string;
}

export interface BoundingBox {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface FigureElement {
  id: string;
  type: 'AXIS' | 'LEGEND' | 'CHART_REGION' | 'LABEL';
  text: string;
  bbox?: number[];
}

export interface Paragraph {
  paragraph_id: string;
  page?: number;
  text: string;
  figure_references?: string[];
}

export interface Section {
  section_id: string;
  title: string;
  paragraphs: Paragraph[];
}

export interface DocumentStructure {
  paper_id: string;
  title: string;
  num_pages?: number;
  sections: Section[];
  figures?: any[];
  metadata?: Record<string, any>;
}

export interface ContextChunk {
  paper_id?: string;
  section_id?: string;
  paragraph_id?: string;
  entity_type?: string;
  chunk_type: 'paper' | 'section' | 'paragraph' | 'entity';
  text: string;
  score?: number;
}

export interface HierarchicalRetrievalResults {
  query: string;
  weighted_score: number;
  level_results: {
    paper: ContextChunk[];
    section: ContextChunk[];
    paragraph: ContextChunk[];
    entity: ContextChunk[];
  };
}

export interface GraphNode {
  id: string;
  label: string;
  node_type: 'FIGURE_ELEMENT' | 'SCIENTIFIC_ENTITY' | 'PAPER_CLAIM';
  element_type?: string;
  entity_type?: EntityType;
  x?: number;
  y?: number;
}

export interface GraphEdge {
  source: string;
  target: string;
  relation: string;
  /** "gnn" on links predicted by the supervised GNN */
  link_source?: string;
  gnn_grounding_score?: number;
}

export interface GroundingGraphSummary {
  num_nodes: number;
  num_edges: number;
  nodes: GraphNode[];
  edges: GraphEdge[];
  gnn_enhanced?: boolean;
  gnn_metrics?: Record<string, any>;
  status: string;
}

export interface ClaimVerification {
  claim: string;
  evidence_snippet: string;
  status: 'SUPPORTED' | 'CONTRADICTED' | 'UNKNOWN';
}

export interface SCQEvaluation {
  scq_composite_score: number;
  breakdown: {
    entity_coverage: number;
    technical_correctness: number;
    reconstruction_fidelity: number;
    retrieval_accuracy: number;
  };
  weights_used: Record<string, number>;
}

export interface ThreeDBranchAnalysis {
  "3d_detection": {
    is_3d: boolean;
    visualization_type: string;
    confidence: number;
    detected_cues: string[];
  };
  raw_geometry_available?: boolean;
  image_to_3d_embedding_dim?: number;
  pointnet_embedding_norm?: number;
  multiview_embedding_norm?: number;
  status: string;
}

export interface PipelineResult {
  paper_id: string;
  figure_id: string;
  query: string;
  extracted_entities: Entity[];
  retrieved_context: HierarchicalRetrievalResults;
  figure_elements: {
    figure_id: string;
    dimensions?: { width: number; height: number };
    status: string;
    elements: FigureElement[];
  };
  grounding_graph_summary: GroundingGraphSummary;
  generated_caption: string;
  verification_results: {
    claims_verification: ClaimVerification[];
    technical_correctness_ratio: number;
    status: string;
  };
  scq_evaluation: SCQEvaluation;
  "3d_branch_analysis"?: ThreeDBranchAnalysis;
  status_declarations: Record<string, string>;
  /** path of the analysed figure; served by GET /figure-image */
  figure_path?: string;
  /** document hierarchy the pipeline actually used */
  document_structure?: DocumentStructure;
  /** true when the UI is showing built-in demo data instead of a backend result */
  is_mock_data?: boolean;
}
