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

const GARTLAND_REFERENCE = [
  { grade: "Normal",    color: "bg-green-500",  description: "No fracture" },
  { grade: "Grade 1",   color: "bg-sky-500",    description: "Undisplaced — AHL transects capitellum middle third" },
  { grade: "Grade 2a",  color: "bg-amber-500",  description: "Posterior displacement, no rotation — AHL anterior to capitellum" },
  { grade: "Grade 2b",  color: "bg-orange-500", description: "Grade 2 + rotational malalignment" },
  { grade: "Grade 3",   color: "bg-red-500",    description: "Complete displacement" },
];

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

      {/* Gartland reference */}
      <details className="rounded-lg border border-border">
        <summary className="cursor-pointer px-4 py-2 text-sm font-medium hover:bg-muted select-none">
          Gartland Classification Reference
        </summary>
        <div className="px-4 py-3 space-y-2">
          {GARTLAND_REFERENCE.map(({ grade: g, color, description }) => (
            <div key={g} className="flex items-start gap-3">
              <span className={`mt-1 inline-block w-3 h-3 rounded-full flex-shrink-0 ${color}`} />
              <div>
                <span className="text-sm font-semibold">{g}</span>
                <span className="text-sm text-muted-foreground"> — {description}</span>
              </div>
            </div>
          ))}
          <p className="text-xs text-muted-foreground pt-2">
            AHL = Anterior Humeral Line drawn along the anterior cortex of the humerus on the lateral X-ray.
            In a normal elbow it transects the middle third of the capitellum.
          </p>
        </div>
      </details>

      {/* Pipeline log */}
      {result.log.length > 0 && (
        <details className="rounded-lg border border-border">
          <summary className="cursor-pointer px-4 py-2 text-sm font-medium hover:bg-muted select-none">
            Technical log ({result.log.length} lines)
          </summary>
          <div className="px-4 py-3 bg-slate-50 rounded-b-lg max-h-64 overflow-y-auto">
            {result.log.map((line, i) => (
              <p key={i} className="text-xs font-mono text-slate-600 leading-relaxed">
                {line}
              </p>
            ))}
          </div>
        </details>
      )}
    </div>
  );
}
