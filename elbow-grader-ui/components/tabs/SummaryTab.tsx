import { Card, CardContent } from "@/components/ui/card";
import type { PredictResponse } from "@/lib/types";

interface SummaryTabProps {
  result: PredictResponse;
}

const GRADE_COLORS: Record<string, { badge: string; card: string }> = {
  Normal:     { badge: "bg-green-100 text-green-800 border-green-200", card: "border-green-200 bg-green-50" },
  "Grade 1":  { badge: "bg-sky-100 text-sky-800 border-sky-200",       card: "border-sky-200 bg-sky-50" },
  "Grade 2a": { badge: "bg-amber-100 text-amber-800 border-amber-200", card: "border-amber-200 bg-amber-50" },
  "Grade 2b": { badge: "bg-orange-100 text-orange-800 border-orange-200", card: "border-orange-200 bg-orange-50" },
  "Grade 3":  { badge: "bg-red-100 text-red-800 border-red-200",       card: "border-red-200 bg-red-50" },
};

function MetricCard({ label, value, colorClass }: { label: string; value: string; colorClass?: string }) {
  return (
    <Card className={colorClass}>
      <CardContent className="pt-4 pb-4">
        <p className="text-xs text-muted-foreground font-medium uppercase tracking-wide mb-1">
          {label}
        </p>
        <p className="text-2xl font-bold">{value}</p>
      </CardContent>
    </Card>
  );
}

export function SummaryTab({ result }: SummaryTabProps) {
  const grade = result.final_grade ?? "UNKNOWN";
  const gradeColor = GRADE_COLORS[grade];
  const isDiscordant = result.discordant;

  return (
    <div className="space-y-6 py-4">
      {/* Grade + confidence row */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <MetricCard
          label="Gartland Grade"
          value={grade}
          colorClass={gradeColor?.card}
        />
        <MetricCard
          label="CNN"
          value={result.cnn_grade ?? "n/a"}
        />
        <MetricCard
          label="Bone geometry"
          value={result.geometric_grade ?? "n/a"}
        />
        <MetricCard
          label="Agreement"
          value={isDiscordant ? "Disagree ⚠️" : "Agree ✓"}
          colorClass={isDiscordant ? "border-amber-200 bg-amber-50" : undefined}
        />
      </div>

      {isDiscordant && (
        <div className="flex items-start gap-3 rounded-lg bg-amber-50 border border-amber-200 px-4 py-3 text-sm text-amber-800">
          <span className="flex-shrink-0 text-base">⚠️</span>
          <span>
            The image AI and bone geometry analyses gave different results. The final grade uses{" "}
            <strong>{result.grade_source}</strong> as the deciding track. Please review the AI Steps
            and AHL tabs for more detail.
          </span>
        </div>
      )}

    </div>
  );
}
