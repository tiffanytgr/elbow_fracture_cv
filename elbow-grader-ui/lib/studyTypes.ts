// Types and constants for the reader study workflow (control vs AI arm).

export type StudyMode = "ai" | "control";

/**
 * Gartland grade options the reader can pick. `value` is stored in the log and
 * matches the AI pipeline's `final_grade` label space so reader and AI answers
 * line up directly; `label` is the compact button text.
 */
export const GRADE_OPTIONS: { value: string; label: string }[] = [
  { value: "Normal", label: "Normal" },
  { value: "Grade 1", label: "1" },
  { value: "Grade 2a", label: "2a" },
  { value: "Grade 2b", label: "2b" },
  { value: "Grade 3", label: "3" },
];

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

/** One reader answer: grade + confidence. */
export interface ReaderAnswer {
  grade: string | null;
  confidence: ConfidenceLevel | null;
}
