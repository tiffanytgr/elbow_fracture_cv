// TypeScript types mirroring the FastAPI /predict response schema.

export interface ExperimentResult {
  labels: string[];
  probs: number[];
  pred_idx: number;
  skipped_reason: string | null;
  ood_score: number | null;
  ood_flagged: boolean;
}

export interface BaumannData {
  baumann_angle_deg: number | null;
  in_normal_range: boolean | null;
  status: string;
  shaft_angle_deg: number | null;
  physis_angle_deg: number | null;
  physis_confidence: number | null;
}

export interface AhlDiagnostic {
  zone?: string;
  method?: string;
  ahl_x_at_cap?: number;
  dist_to_ahl_px?: number;
  cap_radius_px?: number;
  split_pct_pos?: number;
  split_pct_neg?: number;
  bisection_quality_pct?: number;
}

export interface WidthProfileSummary {
  height_px: number | null;
  match_ratio: number | null;
}

export interface GeometricData {
  grade_1v2: string | null;
  grade_2ab: string | null;
  final_grade: string | null;
  bone_sam_score: number | null;
  skipped_reason: string | null;
  ahl_diagnostic: AhlDiagnostic;
  width_profile: WidthProfileSummary | null;
}

export interface PredictResponse {
  final_grade: string | null;
  cnn_grade: string | null;
  geometric_grade: string | null;
  grade_source: string | null;
  discordant: boolean;
  confidence: number | null;
  is_ood: boolean;
  baumann_angle: number | null;
  baumann_normal: boolean | null;
  log: string[];
  ap_sha1: string | null;
  lat_sha1: string | null;
  config_snapshot: Record<string, unknown>;
  result_json: Record<string, unknown>;
  experiments: {
    exp1: ExperimentResult | null;
    exp2: ExperimentResult | null;
    exp3: ExperimentResult | null;
    exp4: ExperimentResult | null;
  };
  baumann: BaumannData | null;
  geometric: GeometricData | null;
  /** Keys: gradcam_1..4, geometric, cortical_width, baumann — base64 PNG */
  plots: Record<string, string>;
  model_status: Record<string, string>;
}

export interface HealthResponse {
  status: string;
  model_status: Record<string, string>;
}

// Form state managed on the client
export interface GraderConfig {
  confidenceThreshold: number;
  runFullLatAlignment: boolean;
  runSam2: boolean;
}
