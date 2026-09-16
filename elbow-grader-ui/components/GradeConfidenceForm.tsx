"use client";

import {
  CONFIDENCE_LABELS,
  CONFIDENCE_LEVELS,
  GRADE_OPTIONS,
  type ConfidenceLevel,
} from "@/lib/studyTypes";

interface GradeConfidenceFormProps {
  grade: string | null;
  confidence: ConfidenceLevel | null;
  onGradeChange: (grade: string) => void;
  onConfidenceChange: (confidence: ConfidenceLevel) => void;
  /** When true, the inputs render read-only (e.g. a locked pre-AI answer). */
  disabled?: boolean;
  idPrefix: string;
}

/** Reusable Gartland grade + 1–5 confidence picker used for pre- and post-AI reads. */
export function GradeConfidenceForm({
  grade,
  confidence,
  onGradeChange,
  onConfidenceChange,
  disabled = false,
  idPrefix,
}: GradeConfidenceFormProps) {
  return (
    <div className="space-y-4">
      <div>
        <span className="mb-1.5 block text-sm font-medium text-slate-700">
          Gartland grade
        </span>
        <div
          role="group"
          aria-label="Gartland grade"
          className="flex flex-wrap gap-2"
        >
          {GRADE_OPTIONS.map((g) => {
            const selected = grade === g.value;
            return (
              <button
                key={g.value}
                type="button"
                id={`${idPrefix}-grade-${g.value.replace(/\s+/g, "-")}`}
                disabled={disabled}
                aria-pressed={selected}
                onClick={() => onGradeChange(g.value)}
                className={`min-w-[3.25rem] rounded-lg border px-4 py-2 text-sm font-semibold transition disabled:cursor-not-allowed disabled:opacity-60 ${
                  selected
                    ? "border-blue-600 bg-blue-600 text-white shadow-sm"
                    : "border-slate-300 bg-white text-slate-700 hover:border-blue-300"
                }`}
              >
                {g.label}
              </button>
            );
          })}
        </div>
      </div>

      <div>
        <span className="mb-1.5 block text-sm font-medium text-slate-700">
          Confidence
        </span>
        <div
          role="group"
          aria-label="Confidence"
          className="flex flex-wrap items-center gap-2"
        >
          {CONFIDENCE_LEVELS.map((level) => {
            const selected = confidence === level;
            return (
              <button
                key={level}
                type="button"
                id={`${idPrefix}-conf-${level}`}
                disabled={disabled}
                aria-pressed={selected}
                title={CONFIDENCE_LABELS[level]}
                onClick={() => onConfidenceChange(level)}
                className={`flex h-10 w-10 items-center justify-center rounded-full border text-sm font-semibold transition disabled:cursor-not-allowed disabled:opacity-60 ${
                  selected
                    ? "border-blue-600 bg-blue-600 text-white shadow-sm"
                    : "border-slate-300 bg-white text-slate-700 hover:border-blue-300"
                }`}
              >
                {level}
              </button>
            );
          })}
          <span className="ml-1 text-xs text-slate-500">
            {confidence
              ? CONFIDENCE_LABELS[confidence]
              : "1 = very unsure · 5 = very confident"}
          </span>
        </div>
      </div>
    </div>
  );
}
