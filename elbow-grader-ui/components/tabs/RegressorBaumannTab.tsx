import { CheckCircle2, AlertTriangle, Info } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import type { PredictResponse } from "@/lib/types";

interface RegressorBaumannTabProps {
  result: PredictResponse;
}

function MetricCard({ label, value }: { label: string; value: string }) {
  return (
    <Card>
      <CardContent className="pt-4 pb-4">
        <p className="text-xs text-muted-foreground font-medium uppercase tracking-wide mb-1">
          {label}
        </p>
        <p className="text-xl font-bold">{value}</p>
      </CardContent>
    </Card>
  );
}

export function RegressorBaumannTab({ result }: RegressorBaumannTabProps) {
  const rb = result.regressor_baumann;

  if (!rb) {
    return (
      <div className="py-4 flex items-center gap-2 text-sm text-muted-foreground">
        <Info className="w-4 h-4" />
        Regressor Baumann angle not available — provide an AP image to enable this.
      </div>
    );
  }

  const { baumann_angle_deg, in_normal_range, keypoints } = rb;
  const regressorPlot = result.plots["baumann_regressor"];

  return (
    <div className="py-4 space-y-6">
      <div>
        <h3 className="text-lg font-semibold mb-1">
          AP Explainability — Baumann Angle (Deep Learning Regressor)
        </h3>
        <p className="text-sm text-muted-foreground">
          A <strong>ResNet-18 keypoint regressor</strong> predicts four anatomical landmarks on the
          AP X-ray and computes the Baumann angle from the predicted shaft axis and physis line.
          Normal range: <strong>60–84°</strong> (mean 72°).
        </p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <MetricCard
          label="Baumann angle (regressor)"
          value={baumann_angle_deg != null ? `${baumann_angle_deg.toFixed(1)}°` : "n/a"}
        />
        <MetricCard
          label="In normal range"
          value={
            in_normal_range === true
              ? "Yes (60–84°)"
              : in_normal_range === false
              ? "No"
              : "n/a"
          }
        />
        <MetricCard
          label="Method"
          value="ResNet-18 keypoint"
        />
      </div>

      {baumann_angle_deg != null && (
        <div
          className={`flex items-start gap-3 rounded-lg px-4 py-3 border ${
            in_normal_range
              ? "bg-green-50 border-green-200"
              : "bg-amber-50 border-amber-200"
          }`}
        >
          {in_normal_range ? (
            <CheckCircle2 className="w-5 h-5 text-green-600 mt-0.5 flex-shrink-0" />
          ) : (
            <AlertTriangle className="w-5 h-5 text-amber-600 mt-0.5 flex-shrink-0" />
          )}
          <p className={`text-sm ${in_normal_range ? "text-green-800" : "text-amber-800"}`}>
            {in_normal_range
              ? `Regressor Baumann angle ${baumann_angle_deg.toFixed(1)}° is within the normal range (60–84°).`
              : `Regressor Baumann angle ${baumann_angle_deg.toFixed(1)}° is outside the normal range (60–84°) — may indicate varus malalignment.`}
          </p>
        </div>
      )}

      {/* Keypoint coordinates */}
      {keypoints && keypoints.length === 4 && (
        <div>
          <h4 className="text-sm font-semibold mb-2">Predicted keypoints (pixel coordinates)</h4>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            {[
              "Prox shaft mid",
              "Dist shaft mid",
              "Med physis",
              "Lat physis",
            ].map((name, i) => (
              <Card key={name}>
                <CardContent className="pt-3 pb-3">
                  <p className="text-xs text-muted-foreground font-medium mb-0.5">
                    {name}
                  </p>
                  <p className="text-sm font-mono">
                    ({keypoints[i][0].toFixed(1)}, {keypoints[i][1].toFixed(1)})
                  </p>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      )}

      <div>
        <h4 className="text-sm font-semibold mb-2">Keypoint overlay visualisation</h4>
        {regressorPlot ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={`data:image/png;base64,${regressorPlot}`}
            alt="Regressor Baumann angle keypoint overlay"
            className="rounded-md max-w-full"
          />
        ) : (
          <p className="text-sm text-muted-foreground italic">No visualisation available.</p>
        )}
      </div>

      <details className="rounded-lg border border-border">
        <summary className="cursor-pointer px-4 py-2 text-sm font-medium hover:bg-muted select-none">
          How does the keypoint regressor work?
        </summary>
        <div className="px-4 py-3 text-sm text-muted-foreground space-y-2">
          <p>
            The regressor is a <strong>ResNet-18</strong> model trained on manually annotated AP
            elbow X-rays. It predicts four keypoints:
          </p>
          <ol className="list-decimal pl-5 space-y-1">
            <li><strong>Proximal shaft midpoint</strong> — midpoint between cortices in the upper shaft</li>
            <li><strong>Distal shaft midpoint</strong> — midpoint between cortices above the olecranon fossa</li>
            <li><strong>Medial physis endpoint</strong> — where the capitellar growth plate meets the metaphysis</li>
            <li><strong>Lateral physis endpoint</strong> — lateral edge of the capitellar growth plate</li>
          </ol>
          <p className="font-medium">Angle computation:</p>
          <ul className="list-disc pl-5 space-y-1">
            <li><strong>Shaft axis</strong> = vector from proximal → distal shaft midpoint</li>
            <li><strong>Physis line</strong> = vector from medial → lateral physis endpoint</li>
            <li><strong>Baumann angle</strong> = acute angle between shaft axis and physis line</li>
          </ul>
          <p>
            This approach is more robust than traditional image processing (Hough lines) because the
            deep learning model learns to localise landmarks even in the presence of fractures,
            overlapping hardware, or poor contrast.
          </p>
        </div>
      </details>
    </div>
  );
}
