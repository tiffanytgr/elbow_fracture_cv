// Types for the clinician control app (manual grading, no AI).

/** Modified Gartland classification options offered to the clinician. */
export const GARTLAND_GRADES = [
  "Normal",
  "I",
  "IIA",
  "IIB",
  "III",
  "IV",
] as const;

export type GartlandGrade = (typeof GARTLAND_GRADES)[number];

/** 1–5 confidence (Likert) scale. */
export const CONFIDENCE_LEVELS = [1, 2, 3, 4, 5] as const;
export type ConfidenceLevel = (typeof CONFIDENCE_LEVELS)[number];

export const CONFIDENCE_LABELS: Record<ConfidenceLevel, string> = {
  1: "Very unsure",
  2: "Unsure",
  3: "Moderate",
  4: "Confident",
  5: "Very confident",
};

/** One saved clinician assessment, mirroring the JSONL log record. */
export interface AssessmentRecord {
  case_id: string | null;
  clinician: string | null;
  grade: GartlandGrade | null;
  confidence: ConfidenceLevel | null;
  notes: string | null;
  input_mode: string | null;
  elapsed_seconds: number;
  elapsed_hms: string;
  started_at: string | null;
  ended_at: string | null;
  logged_at: string;
}
