import { CheckCircle2, AlertTriangle, Info } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import type { PredictResponse } from "@/lib/types";

interface BaumannTabProps {
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

const STATUS_MESSAGES: Record<string, string> = {
  no_yolo_ckpt:
    "YOLO humerus model checkpoint not found on the server. Contact the administrator.",
  no_mask:
    "Humerus was not detected in this image — check image quality or field of view.",
  no_shaft:
    "Could not fit a shaft axis — ensure the full humerus shaft is visible.",
  no_physis:
    "Could not fit the physeal line — check image quality and distal humerus visibility.",
};

export function BaumannTab({ result }: BaumannTabProps) {
  const b = result.baumann;

  if (!b) {
    return (
      <div className="py-4 flex items-center gap-2 text-sm text-muted-foreground">
        <Info className="w-4 h-4" />
        Baumann angle not available — provide an AP image to enable this.
      </div>
    );
  }

  const { baumann_angle_deg, in_normal_range, status, shaft_angle_deg,
          physis_angle_deg, physis_confidence } = b;

  if (status !== "ok") {
    const msg =
      STATUS_MESSAGES[status] ??
      (status.startsWith("error:")
        ? `Pipeline error: ${status.replace("error:", "").trim()}`
        : `Baumann pipeline could not complete (status: ${status}).`);
    return (
      <div className="py-4 space-y-3">
        <div className="flex items-start gap-2 text-sm text-amber-700 bg-amber-50 border border-amber-200 rounded-lg px-4 py-3">
          <AlertTriangle className="w-4 h-4 mt-0.5 flex-shrink-0" />
          <span>{msg}</span>
        </div>
        <p className="text-xs text-muted-foreground">
          Pipeline status: <code>{status}</code>
        </p>
      </div>
    );
  }
  const baumannPlot = result.plots["baumann"];

  return (
    <div className="py-4 space-y-6">
      <div>
        <h3 className="text-lg font-semibold mb-1">Baumann Angle (AP view)</h3>
        <p className="text-sm text-muted-foreground">
          The <strong>Baumann angle</strong> is the angle between the humeral shaft axis
          and the capitellar physis line on the AP X-ray. Normal range:{" "}
          <strong>60–84°</strong> (mean 72°). A reduced angle indicates varus alignment
          from a supracondylar fracture.
        </p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <MetricCard
          label="Baumann angle"
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
        <MetricCard label="Pipeline status" value={status} />
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
              ? `Baumann angle ${baumann_angle_deg.toFixed(1)}° is within the normal range (60–84°).`
              : `Baumann angle ${baumann_angle_deg.toFixed(1)}° is outside the normal range (60–84°) — may indicate varus malalignment.`}
          </p>
        </div>
      )}

      {shaft_angle_deg != null && (
        <p className="text-xs text-muted-foreground">
          Shaft axis: {shaft_angle_deg.toFixed(1)}° from vertical
          {physis_angle_deg != null && ` | Physis angle: ${physis_angle_deg.toFixed(1)}° from horizontal`}
          {physis_confidence != null && ` | Physis detection confidence: ${physis_confidence.toFixed(2)}`}
        </p>
      )}

      <div>
        <h4 className="text-sm font-semibold mb-2">Measurement visualisation</h4>
        {baumannPlot ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={`data:image/png;base64,${baumannPlot}`}
            alt="Baumann angle visualisation"
            className="rounded-md max-w-full"
          />
        ) : (
          <p className="text-sm text-muted-foreground italic">No visualisation available.</p>
        )}
      </div>

      <details className="rounded-lg border border-border">
        <summary className="cursor-pointer px-4 py-2 text-sm font-medium hover:bg-muted select-none">
          What is the Baumann angle?
        </summary>
        <div className="px-4 py-3 text-sm text-muted-foreground space-y-2">
          <p>
            The <strong>Baumann angle</strong> (capitellar-humeral angle) measures the orientation
            of the distal humeral physis relative to the shaft:
          </p>
          <ul className="list-disc pl-5 space-y-1">
            <li><strong>Normal</strong>: 60–84° (literature mean 72°)</li>
            <li>
              <strong>Decreased angle (&lt; 60°)</strong>: suggests varus malalignment — the distal
              fragment has tilted medially relative to the shaft
            </li>
            <li>
              <strong>Increased angle (&gt; 84°)</strong>: suggests valgus malalignment
            </li>
          </ul>
          <p>
            In supracondylar fractures, the Baumann angle is used to assess reduction quality and
            guide surgical planning.
          </p>
          <p className="font-medium">Measurement pipeline:</p>
          <ol className="list-decimal pl-5 space-y-1">
            <li>YOLO instance segmentation → humerus mask (highest-confidence detection)</li>
            <li>PCA on mask pixels → rotate image so shaft axis is vertical (upright alignment)</li>
            <li>Peak bone-width row → condyle level; PCA on proximal midpoints → shaft axis</li>
            <li>Sigma-clipped linear fit to the distal boundary of the humerus mask → physeal line</li>
            <li>Baumann angle = acute angle between shaft axis and physeal line</li>
          </ol>
        </div>
      </details>
    </div>
  );
}
