import { AlertTriangle, CheckCircle2, XCircle, Info } from "lucide-react";
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

export function ResultsBanner({ result, hasLat }: ResultsBannerProps) {
  const { final_grade, grade_source, discordant, cnn_grade, geometric_grade } = result;

  const exp1 = result.experiments.exp1;
  const fractureDetectedNoLat =
    !hasLat &&
    exp1 !== null &&
    exp1.pred_idx === 1 &&
    (final_grade === null || final_grade === "UNKNOWN");

  const style = final_grade ? GRADE_STYLES[final_grade] : null;

  let gradeCard: React.ReactNode;

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
        <Icon className={`w-5 h-5 flex-shrink-0 ${style.iconClass}`} />
        <span className={`text-base font-bold ${style.text}`}>{final_grade}</span>
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
    <div className="space-y-3">
      <h2 className="text-base font-semibold">Classification Result</h2>
      {gradeCard}

      {/* Supporting AI tracks */}
      {(cnn_grade || geometric_grade) && (
        <div className="flex gap-2 flex-wrap">
          <div className="flex items-center gap-1.5 rounded-md bg-blue-50 border border-blue-200 px-2.5 py-1.5 text-xs">
            <span>🧠</span>
            <span className="font-medium text-blue-800">CNN</span>
            <Badge variant="info" className="text-xs py-0">{cnn_grade ?? "n/a"}</Badge>
          </div>
          <div className="flex items-center gap-1.5 rounded-md bg-purple-50 border border-purple-200 px-2.5 py-1.5 text-xs">
            <span>📐</span>
            <span className="font-medium text-purple-800">Bone geometry</span>
            <Badge variant="info" className="text-xs py-0">{geometric_grade ?? "n/a"}</Badge>
          </div>
        </div>
      )}
    </div>
  );
}
