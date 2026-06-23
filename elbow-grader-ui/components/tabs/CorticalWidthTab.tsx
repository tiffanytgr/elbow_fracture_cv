import { Info, CheckCircle2, AlertTriangle } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import type { PredictResponse } from "@/lib/types";

interface CorticalWidthTabProps {
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

export function CorticalWidthTab({ result }: CorticalWidthTabProps) {
  const g = result.geometric;
  const widthPlot = result.plots["cortical_width"];

  if (!g?.width_profile) {
    return (
      <div className="py-4 flex items-center gap-2 text-sm text-muted-foreground">
        <Info className="w-4 h-4" />
        No cortical width data — LAT geometric track did not run or did not reach Grade 2.
      </div>
    );
  }

  const { height_px, match_ratio } = g.width_profile;
  const grade = g.grade_2ab ?? "n/a";
  const mr = match_ratio ?? 0;
  const isUniform = mr >= 0.8;
  const uniformityLabel = mr >= 0.8 ? "High" : mr >= 0.5 ? "Moderate" : "Low";

  return (
    <div className="py-4 space-y-6">
      <div>
        <h3 className="text-lg font-semibold mb-1">How bone shape distinguishes Grade 2a from 2b</h3>
        <p className="text-sm text-muted-foreground">
          We measure the <strong>width of the humerus bone</strong> at 40 points within{" "}
          <strong>{height_px ?? 30} pixels above the capitellum</strong> and check how{" "}
          <strong>uniform</strong> the width profile is. A uniform profile suggests{" "}
          <strong>Grade 2a</strong>, while an uneven profile suggests <strong>Grade 2b</strong> —
          consistent with rotational displacement.
        </p>
      </div>

      {widthPlot && (
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={`data:image/png;base64,${widthPlot}`}
          alt="Cortical width profile"
          className="rounded-md max-w-full"
        />
      )}

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <MetricCard label="Predicted sub-grade" value={grade} />
        <MetricCard
          label="Match ratio"
          value={mr.toFixed(2)}
          sub="1.0 = uniform, 0.0 = extremely uneven"
        />
        <MetricCard label="Uniformity" value={uniformityLabel} />
      </div>

      <div className="space-y-2">
        <p className="text-sm font-semibold">Width uniformity</p>
        <Progress value={Math.min(Math.round(mr * 100), 100)} className="h-4" />
      </div>

      <div
        className={`flex items-start gap-3 rounded-lg px-4 py-3 border ${
          isUniform ? "bg-green-50 border-green-200" : "bg-amber-50 border-amber-200"
        }`}
      >
        {isUniform ? (
          <CheckCircle2 className="w-5 h-5 text-green-600 mt-0.5 flex-shrink-0" />
        ) : (
          <AlertTriangle className="w-5 h-5 text-amber-600 mt-0.5 flex-shrink-0" />
        )}
        <p className={`text-sm ${isUniform ? "text-green-800" : "text-amber-800"}`}>
          {isUniform
            ? `Match ratio = ${mr.toFixed(2)} — uniform width profile → Grade 2a`
            : `Match ratio = ${mr.toFixed(2)} — non-uniform profile → Grade 2b`}
        </p>
      </div>

      <details className="rounded-lg border border-border">
        <summary className="cursor-pointer px-4 py-2 text-sm font-medium hover:bg-muted select-none">
          What does this mean?
        </summary>
        <div className="px-4 py-3 text-sm text-muted-foreground space-y-2">
          <p>
            <strong>Match ratio</strong> compares bone width between the lower and upper halves of
            the measurement region (near the joint vs. further up the shaft):
          </p>
          <ul className="list-disc pl-5 space-y-1">
            <li>
              <strong>≥ 0.80</strong> → even diameter → <strong>Grade 2a</strong> (no significant
              rotational displacement)
            </li>
            <li>
              <strong>&lt; 0.80</strong> → narrower near joint → <strong>Grade 2b</strong>{" "}
              (rotational displacement)
            </li>
          </ul>
          <p className="font-medium">Statistical validation (199 images after outlier removal):</p>
          <ul className="list-disc pl-5 space-y-1">
            <li>Grade 2a mean match ratio: <strong>0.59</strong> (n=143)</li>
            <li>Grade 2b mean match ratio: <strong>0.47</strong> (n=56)</li>
            <li>
              Welch&apos;s t-test: <strong>p = 0.020</strong> | Mann-Whitney U:{" "}
              <strong>p = 0.036</strong> | Permutation: <strong>p = 0.011</strong>
            </li>
          </ul>
        </div>
      </details>
    </div>
  );
}
