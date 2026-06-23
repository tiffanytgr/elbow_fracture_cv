import { Info, CheckCircle2, XCircle } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import type { PredictResponse } from "@/lib/types";

interface GeometricTabProps {
  result: PredictResponse;
}

function MetricCard({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <Card>
      <CardContent className="pt-4 pb-4">
        <p className="text-xs text-muted-foreground font-medium uppercase tracking-wide mb-1">
          {label}
        </p>
        <p className="text-xl font-bold">{value}</p>
        {sub && <p className="text-xs text-muted-foreground mt-0.5">{sub}</p>}
      </CardContent>
    </Card>
  );
}

export function GeometricTab({ result }: GeometricTabProps) {
  const g = result.geometric;
  const overlayPlot = result.plots["geometric"];

  if (!g || g.skipped_reason) {
    return (
      <div className="py-4 flex items-center gap-2 text-sm text-muted-foreground">
        <Info className="w-4 h-4" />
        Geometric track skipped: {g?.skipped_reason ?? "no LAT input"}
      </div>
    );
  }

  const ahl = g.ahl_diagnostic ?? {};
  const pctR = ahl.split_pct_pos;
  const pctL = ahl.split_pct_neg;
  const bq = ahl.bisection_quality_pct;
  const ahlPass = pctR != null && pctR >= 30 && pctR <= 70;

  return (
    <div className="py-4 space-y-6">
      {/* SAM2 overlay */}
      {overlayPlot && (
        <div>
          <h3 className="text-sm font-semibold mb-2">SAM2 Segmentation Overlay</h3>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={`data:image/png;base64,${overlayPlot}`}
            alt="SAM2 overlay"
            className="rounded-md max-w-xs"
          />
        </div>
      )}

      {/* Metrics */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <MetricCard label="Geometric Grade 1 vs 2" value={g.grade_1v2 ?? "n/a"} />
        <MetricCard label="Geometric Grade 2a vs 2b" value={g.grade_2ab ?? "n/a"} />
        <MetricCard
          label="Bone SAM score"
          value={g.bone_sam_score != null ? g.bone_sam_score.toFixed(3) : "n/a"}
        />
      </div>

      {/* AHL bisection */}
      {pctR != null && pctL != null && (
        <div className="space-y-3">
          <h4 className="text-sm font-semibold">AHL bisection score</h4>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <MetricCard
              label="Right of AHL"
              value={`${pctR.toFixed(1)}%`}
              sub="% of capitellum pixels to the right"
            />
            <MetricCard
              label="Left of AHL"
              value={`${pctL.toFixed(1)}%`}
              sub="% of capitellum pixels to the left"
            />
            <MetricCard
              label="Bisection quality"
              value={bq != null ? `${bq.toFixed(1)}%` : "n/a"}
              sub="2 × min(right%, left%) — 100% = perfect"
            />
          </div>

          {/* Split bar */}
          <div className="space-y-1">
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>← Left of AHL</span>
              <span>Right of AHL →</span>
            </div>
            <Progress value={Math.round(pctR)} className="h-4" />
          </div>

          <div
            className={`flex items-start gap-3 rounded-lg px-4 py-3 border ${
              ahlPass ? "bg-green-50 border-green-200" : "bg-red-50 border-red-200"
            }`}
          >
            {ahlPass ? (
              <CheckCircle2 className="w-5 h-5 text-green-600 mt-0.5 flex-shrink-0" />
            ) : (
              <XCircle className="w-5 h-5 text-red-600 mt-0.5 flex-shrink-0" />
            )}
            <p className={`text-sm ${ahlPass ? "text-green-800" : "text-red-800"}`}>
              {ahlPass
                ? `Split ${pctR.toFixed(1)}% / ${pctL.toFixed(1)}% — within 30–70% threshold (supports Grade 1)`
                : `Split ${pctR.toFixed(1)}% / ${pctL.toFixed(1)}% — outside 30–70% threshold → Grade 2`}
            </p>
          </div>
        </div>
      )}

      {/* AHL details */}
      {Object.keys(ahl).length > 0 && (
        <div>
          <h4 className="text-sm font-semibold mb-2">AHL details</h4>
          <pre className="text-xs bg-slate-50 rounded-lg border border-border p-3 overflow-x-auto">
            {JSON.stringify(
              Object.fromEntries(
                Object.entries(ahl).filter(([k]) =>
                  ["zone", "method", "ahl_x_at_cap", "dist_to_ahl_px", "cap_radius_px"].includes(k),
                ),
              ),
              null,
              2,
            )}
          </pre>
        </div>
      )}
    </div>
  );
}
