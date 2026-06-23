import Image from "next/image";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import type { ExperimentResult, PredictResponse } from "@/lib/types";

interface PerExperimentTabProps {
  result: PredictResponse;
}

const EXP_LABELS: Record<string, string> = {
  exp1: "Exp 1 — Normal vs Fractured (AP)",
  exp2: "Exp 2 — Grade 3 vs Grade 1/2 (AP)",
  exp3: "Exp 3 — Grade 1 vs Grade 2 (LAT)",
  exp4: "Exp 4 — Grade 2a vs Grade 2b (LAT)",
};

function ExperimentPanel({
  expKey,
  exp,
  gradcamB64,
}: {
  expKey: string;
  exp: ExperimentResult | null;
  gradcamB64?: string;
}) {
  const label = EXP_LABELS[expKey] ?? expKey;

  if (exp === null) {
    return (
      <div className="rounded-lg border border-border p-3 text-sm text-muted-foreground italic">
        {label}: not run
      </div>
    );
  }

  if (exp.skipped_reason) {
    const isLowConf = exp.skipped_reason.startsWith("low_confidence");
    return (
      <div className="rounded-lg border border-border p-4 space-y-3">
        <div className="flex items-center gap-2">
          <span className="font-semibold text-sm">{label}</span>
          <Badge variant={isLowConf ? "warning" : "secondary"}>
            {isLowConf ? "Low confidence" : "Skipped"}
          </Badge>
        </div>
        {isLowConf && exp.probs && exp.probs.length > 0 && (
          <ProbBars labels={exp.labels} probs={exp.probs} predIdx={-1} />
        )}
      </div>
    );
  }

  const predLabel = exp.pred_idx >= 0 && exp.pred_idx < exp.labels.length
    ? exp.labels[exp.pred_idx]
    : "?";

  return (
    <div className="rounded-lg border border-border p-4 space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <span className="font-semibold text-sm">{label}</span>
        <Badge variant="info">{predLabel}</Badge>
        {exp.ood_flagged && (
          <Badge variant="warning">OOD {exp.ood_score?.toFixed(4)}</Badge>
        )}
      </div>

      {exp.probs && exp.probs.length > 0 && (
        <ProbBars labels={exp.labels} probs={exp.probs} predIdx={exp.pred_idx} />
      )}

      {gradcamB64 && (
        <div className="mt-2">
          <p className="text-xs text-muted-foreground mb-1">Grad-CAM</p>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={`data:image/png;base64,${gradcamB64}`}
            alt={`Grad-CAM ${expKey}`}
            className="rounded-md max-w-full"
          />
        </div>
      )}
    </div>
  );
}

function ProbBars({
  labels,
  probs,
  predIdx,
}: {
  labels: string[];
  probs: number[];
  predIdx: number;
}) {
  return (
    <div className="space-y-1.5">
      {labels.map((lbl, i) => (
        <div key={i} className="flex items-center gap-2">
          <span className="text-xs w-28 text-right shrink-0 font-medium text-slate-600">{lbl}</span>
          <Progress
            value={Math.round((probs[i] ?? 0) * 100)}
            className={`h-3 flex-1 ${i === predIdx ? "[&>div]:bg-green-500" : "[&>div]:bg-slate-300"}`}
          />
          <span className="text-xs font-mono w-14 text-right text-slate-500">
            {((probs[i] ?? 0) * 100).toFixed(1)}%
          </span>
        </div>
      ))}
    </div>
  );
}

export function PerExperimentTab({ result }: PerExperimentTabProps) {
  const { experiments, plots } = result;

  return (
    <div className="py-4 grid grid-cols-1 xl:grid-cols-2 gap-6">
      <div className="space-y-4">
        <div>
          <h3 className="font-semibold text-sm text-muted-foreground uppercase tracking-wide">
            AP View — CNN Cascade
          </h3>
          <p className="text-xs text-muted-foreground mt-0.5">
            Supported by <span className="font-medium text-slate-600">Baumann Angle (AP)</span> tab
            for geometric validation of the AP classification.
          </p>
        </div>
        <ExperimentPanel expKey="exp1" exp={experiments.exp1} gradcamB64={plots["gradcam_1"]} />
        <ExperimentPanel expKey="exp2" exp={experiments.exp2} gradcamB64={plots["gradcam_2"]} />
      </div>
      <div className="space-y-4">
        <div>
          <h3 className="font-semibold text-sm text-muted-foreground uppercase tracking-wide">
            LAT View — CNN Cascade
          </h3>
          <p className="text-xs text-muted-foreground mt-0.5">
            Supported by <span className="font-medium text-slate-600">AHL Measurement (LAT)</span> tab
            for geometric validation of the lateral classification.
          </p>
        </div>
        <ExperimentPanel expKey="exp3" exp={experiments.exp3} gradcamB64={plots["gradcam_3"]} />
        <ExperimentPanel expKey="exp4" exp={experiments.exp4} gradcamB64={plots["gradcam_4"]} />
      </div>
    </div>
  );
}
