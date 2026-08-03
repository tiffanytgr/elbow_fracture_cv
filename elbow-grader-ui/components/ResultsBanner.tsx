import { AlertTriangle, Check, CheckCircle2, XCircle, Info } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import type { PredictResponse } from "@/lib/types";

interface ResultsBannerProps {
  result: PredictResponse;
  hasLat: boolean;
}

interface GradeStyle {
  bg: string;
  border: string;
  text: string;
  iconClass: string;
  icon: typeof CheckCircle2;
}

const GRADE_STYLES: Record<string, GradeStyle> = {
  Normal:     { bg: "bg-green-50",  border: "border-green-300",  text: "text-green-900",  iconClass: "text-green-600",  icon: CheckCircle2 },
  "Grade 1":  { bg: "bg-sky-50",    border: "border-sky-300",    text: "text-sky-900",    iconClass: "text-sky-600",    icon: CheckCircle2 },
  "Grade 2a": { bg: "bg-amber-50",  border: "border-amber-300",  text: "text-amber-900",  iconClass: "text-amber-600",  icon: AlertTriangle },
  "Grade 2b": { bg: "bg-orange-50", border: "border-orange-300", text: "text-orange-900", iconClass: "text-orange-600", icon: AlertTriangle },
  "Grade 3":  { bg: "bg-red-50",    border: "border-red-300",    text: "text-red-900",    iconClass: "text-red-600",    icon: AlertTriangle },
};

const GARTLAND_REFERENCE = [
  { grade: "Normal", color: "bg-green-500", description: "No fracture" },
  { grade: "Grade 1", color: "bg-sky-500", description: "Undisplaced — AHL transects capitellum middle third" },
  { grade: "Grade 2a", color: "bg-amber-500", description: "Posterior displacement, no rotation — AHL anterior to capitellum" },
  { grade: "Grade 2b", color: "bg-orange-500", description: "Grade 2 + rotational malalignment" },
  { grade: "Grade 3", color: "bg-red-500", description: "Complete displacement" },
];

export function GartlandReference({ currentGrade }: { currentGrade: string | null }) {
  return (
    <details className="overflow-hidden rounded-lg border border-slate-200 bg-slate-50/70 shadow-sm">
      <summary className="cursor-pointer select-none px-4 py-3 text-sm font-semibold text-slate-800 hover:bg-slate-100/70">
        Gartland Classification Reference
      </summary>
      <div className="space-y-3 border-t border-slate-200 p-4">
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-5">
        {GARTLAND_REFERENCE.map(({ grade, color, description }) => (
          <div
            key={grade}
            className={`relative min-h-28 rounded-lg border bg-white p-4 shadow-sm ${
              currentGrade === grade
                ? "border-blue-500 ring-1 ring-blue-200"
                : "border-slate-200"
            }`}
          >
            <div className="flex items-center gap-2 pr-6">
              <span className={`inline-block h-3 w-3 flex-shrink-0 rounded-full ${color}`} />
              <span className="font-semibold text-slate-900">{grade}</span>
            </div>
            <p className="mt-2 pl-5 text-sm leading-5 text-slate-500">
              {description}
            </p>
            {currentGrade === grade && (
              <Check className="absolute right-4 top-4 h-4 w-4 text-blue-600" />
            )}
          </div>
        ))}
      </div>
      <p className="text-xs text-muted-foreground">
        AHL = Anterior Humeral Line drawn along the anterior cortex of the humerus on the lateral X-ray.
        In a normal elbow it transects the middle third of the capitellum.
      </p>
      </div>
    </details>
  );
}

export function ResultsBanner({ result, hasLat }: ResultsBannerProps) {
  const { final_grade, grade_source, discordant } = result;

  const exp1 = result.experiments.exp1;
  const fractureDetectedNoLat =
    !hasLat &&
    exp1 !== null &&
    exp1.pred_idx === 1 &&
    (final_grade === null || final_grade === "UNKNOWN");

  const style = final_grade ? GRADE_STYLES[final_grade] : null;

  let gradeCard: React.ReactNode = null;

  if (fractureDetectedNoLat) {
    gradeCard = (
      <div className="space-y-2">
        <div className="flex items-center gap-3 rounded-lg border-2 border-amber-300 bg-amber-50 px-4 py-3">
          <AlertTriangle className="w-5 h-5 text-amber-600 flex-shrink-0" />
          <span className="text-base font-bold text-amber-900">Fracture detected</span>
          <Badge variant="warning">No LAT uploaded</Badge>
        </div>
        <div className="flex items-center gap-3 rounded-lg bg-blue-50 border border-blue-200 px-4 py-2">
          <Info className="w-4 h-4 text-blue-600 flex-shrink-0" />
          <p className="text-xs text-blue-800">
            Upload the <strong>LAT (lateral) view</strong> to determine the Gartland grade.
          </p>
        </div>
      </div>
    );
  } else if (final_grade === "Grade 1 or 2") {
    gradeCard = (
      <div className="space-y-2">
        <div className="flex items-center gap-3 rounded-lg border-2 border-amber-300 bg-amber-50 px-4 py-3">
          <AlertTriangle className="w-5 h-5 text-amber-600 flex-shrink-0" />
          <span className="text-base font-bold text-amber-900">Grade 1 or 2</span>
          <Badge variant="warning">LAT required</Badge>
        </div>
        <div className="flex items-center gap-3 rounded-lg bg-blue-50 border border-blue-200 px-4 py-2">
          <Info className="w-4 h-4 text-blue-600 flex-shrink-0" />
          <p className="text-xs text-blue-800">Upload the <strong>lateral view</strong> to complete grading.</p>
        </div>
      </div>
    );
  } else if (!final_grade || final_grade === "UNKNOWN") {
    gradeCard = (
      <div className="flex items-center gap-3 rounded-lg border-2 border-red-200 bg-red-50 px-4 py-3">
        <XCircle className="w-5 h-5 text-red-600 flex-shrink-0" />
        <span className="text-base font-bold text-red-900">Unable to grade</span>
        <span className="text-xs text-red-700">Check image quality and retry.</span>
      </div>
    );
  } else if (style) {
    const Icon = style.icon;
    gradeCard = (
      <div className={`flex items-center gap-3 rounded-lg border-2 px-4 py-3 ${style.bg} ${style.border}`}>
        <Icon className={`h-5 w-5 flex-shrink-0 ${style.iconClass}`} />
        <span className={`text-sm font-bold ${style.text}`}>{final_grade}</span>
        <Badge variant={discordant ? "warning" : "success"}>
          {discordant ? "Tracks disagree" : "Tracks agree"}
        </Badge>
        {!discordant && grade_source && (
          <span className={`text-xs opacity-60 ${style.text}`}>{grade_source}</span>
        )}
      </div>
    );
  }

  return (
    <>{gradeCard}</>
  );
}
