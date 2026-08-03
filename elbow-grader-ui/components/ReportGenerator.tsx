"use client";

import { useState } from "react";
import { Download, Loader2 } from "lucide-react";

import { Button } from "@/components/ui/button";
import type { ExperimentResult, PredictResponse } from "@/lib/types";
import {
  getAhlUnavailableMessage,
  getBaumannUnavailableMessage,
  getBoneProfileUnavailableMessage,
} from "@/lib/resultDescriptions";

interface ReportGeneratorProps {
  result: PredictResponse;
  apFile: File | null;
  latFile: File | null;
  confidenceThreshold: number;
}

const GARTLAND = [
  ["Normal", "#22c55e", "No fracture"],
  ["Grade 1", "#0ea5e9", "Undisplaced — AHL transects the capitellum middle third"],
  ["Grade 2a", "#f59e0b", "Posterior displacement, no rotation — AHL anterior to capitellum"],
  ["Grade 2b", "#f97316", "Grade 2 with rotational malalignment"],
  ["Grade 3", "#ef4444", "Complete displacement"],
] as const;

const EXPERIMENT_NAMES = {
  exp1: "Exp 1 — Normal vs Fractured (AP)",
  exp2: "Exp 2 — Grade 3 vs Grade 1/2 (AP)",
  exp3: "Exp 3 — Grade 1 vs Grade 2 (LAT)",
  exp4: "Exp 4 — Grade 2a vs Grade 2b (LAT)",
} as const;

function escapeHtml(value: unknown) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function fileToDataUrl(file: File | null): Promise<string | null> {
  if (!file) return Promise.resolve(null);
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
}

function plotUrl(result: PredictResponse, key: string) {
  const value = result.plots[key];
  return value ? `data:image/png;base64,${value}` : null;
}

function metric(label: string, value: unknown) {
  return `<div class="metric"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value ?? "n/a")}</strong></div>`;
}

function reportImage(src: string | null, alt: string) {
  return src
    ? `<figure><img src="${src}" alt="${escapeHtml(alt)}"><figcaption>${escapeHtml(alt)}</figcaption></figure>`
    : `<p class="unavailable">${escapeHtml(alt)} not available.</p>`;
}

function experimentHtml(
  key: keyof typeof EXPERIMENT_NAMES,
  exp: ExperimentResult | null,
  result: PredictResponse,
) {
  if (!exp) return `<article class="subcard"><h3>${EXPERIMENT_NAMES[key]}</h3><p class="unavailable">Not run.</p></article>`;

  const predicted = exp.pred_idx >= 0 && exp.pred_idx < exp.labels.length
    ? exp.labels[exp.pred_idx]
    : "Not determined";
  const probabilities = exp.labels.map((label, index) => {
    const percent = (exp.probs[index] ?? 0) * 100;
    return `<div class="prob"><span>${escapeHtml(label)}</span><div><i style="width:${Math.max(0, Math.min(100, percent))}%"></i></div><strong>${percent.toFixed(1)}%</strong></div>`;
  }).join("");
  const plotKey = `gradcam_${key.slice(-1)}`;
  const gradcam = plotUrl(result, plotKey);

  return `<article class="subcard">
    <h3>${EXPERIMENT_NAMES[key]}</h3>
    ${metric("Result", exp.skipped_reason ? "Skipped" : predicted)}
    ${exp.skipped_reason ? `<p class="note">Reason: ${escapeHtml(exp.skipped_reason)}</p>` : ""}
    ${probabilities}
    ${exp.ood_flagged ? `<p class="warning">Out-of-distribution flag${exp.ood_score != null ? ` (score ${exp.ood_score.toFixed(4)})` : ""}</p>` : ""}
    ${gradcam ? `<figure><img src="${gradcam}" alt="${escapeHtml(`${EXPERIMENT_NAMES[key]} Grad-CAM`)}"></figure>` : ""}
  </article>`;
}

function buildReportHtml(
  result: PredictResponse,
  apImage: string | null,
  latImage: string | null,
  confidenceThreshold: number,
) {
  const generatedAt = new Intl.DateTimeFormat(undefined, {
    dateStyle: "long",
    timeStyle: "medium",
  }).format(new Date());
  const agreement = result.discordant ? "Disagree" : "Agree";
  const b = result.baumann;
  const g = result.geometric;
  const ahl = g?.ahl_diagnostic ?? {};
  const width = g?.width_profile;
  const baumannUnavailable = getBaumannUnavailableMessage(b?.status);
  const ahlUnavailable = getAhlUnavailableMessage(g);
  const boneProfileUnavailable = getBoneProfileUnavailableMessage(g);
  const matchRatio = width?.match_ratio;
  const uniformity = matchRatio == null ? "n/a" : matchRatio >= 0.8 ? "High" : matchRatio >= 0.5 ? "Moderate" : "Low";

  const reference = GARTLAND.map(([grade, color, description]) => `
    <div class="reference ${result.final_grade === grade ? "selected" : ""}">
      <span class="dot" style="background:${color}"></span><strong>${grade}</strong>
      <p>${description}</p>
    </div>`).join("");

  return `<!doctype html><html><head><meta charset="utf-8"><title>Elbow fracture analysis report</title>
  <style>
    *{box-sizing:border-box;-webkit-print-color-adjust:exact!important;print-color-adjust:exact!important;forced-color-adjust:none} body{margin:0;color:#172033;background:#eef2f7;font:14px/1.45 Arial,sans-serif}
    .toolbar{position:sticky;top:0;z-index:3;display:flex;justify-content:flex-end;gap:10px;padding:12px 24px;background:#0b3474}
    button{border:1px solid #fff;border-radius:7px;padding:9px 14px;background:#fff;color:#0b3474;font-weight:700;cursor:pointer}
    main{width:min(1080px,calc(100% - 32px));margin:24px auto;background:#fff;padding:38px;border-radius:12px;box-shadow:0 4px 20px #16345f18}
    header{padding:24px 28px;border-radius:10px;color:#fff;background:linear-gradient(120deg,#082b68,#0b56a0)}
    h1{margin:0 0 6px;font-size:27px} header p{margin:0;color:#dbeafe}.meta{margin-top:15px;font-size:12px}
    section{margin-top:28px;break-inside:avoid} h2{margin:0 0 12px;padding-bottom:7px;border-bottom:2px solid #dbe7f5;color:#133968;font-size:18px}
    .input-images{margin-top:24px}.reference-block{margin-top:24px;break-inside:avoid}.reference-block h2{font-size:15px}
    h3{margin:0 0 10px;font-size:14px}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}
    .metrics{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}.metrics.three{grid-template-columns:repeat(3,minmax(0,1fr))}.metric,.subcard,.reference{border:1px solid #dbe3ee;border-radius:8px;background:#fff;padding:13px}
    .metric span{display:block;color:#64748b;text-transform:uppercase;font-size:10px;font-weight:700;letter-spacing:.04em}.metric strong{display:block;margin-top:3px;font-size:18px}
    figure{margin:12px 0 0;text-align:center}figure img{display:block;max-width:100%;max-height:520px;margin:auto;object-fit:contain;background:#020617;border-radius:6px}figcaption{margin-top:5px;color:#64748b;font-size:10px}
    .reference-grid{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:8px}.reference{min-height:100px}.reference.selected{border:2px solid #3b82f6;background:#eff6ff}.reference p{margin:7px 0 0 17px;color:#64748b;font-size:11px}.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:7px}
    .prob{display:grid;grid-template-columns:110px 1fr 48px;align-items:center;gap:7px;margin:7px 0;font-size:11px}.prob>div{height:8px;background:#e2e8f0;border-radius:9px;overflow:hidden}.prob i{display:block;height:100%;background:#22c55e}.prob strong{text-align:right}
    .note{padding:9px;border-radius:6px;background:#f1f5f9;color:#475569}.warning{padding:9px;border:1px solid #fcd34d;border-radius:6px;background:#fffbeb;color:#92400e}.unavailable{color:#64748b;font-style:italic}
    .disclaimer{margin-top:30px;padding-top:12px;border-top:1px solid #dbe3ee;color:#64748b;font-size:10px}
    @media(max-width:700px){main{width:100%;margin:0;padding:20px;border-radius:0}.grid,.metrics,.metrics.three,.reference-grid{grid-template-columns:1fr}.toolbar{padding:8px}.reference{min-height:auto}}
    @page{size:A4;margin:12mm}@media print{body{background:#fff}.toolbar{display:none}main{width:auto;margin:0;padding:0;box-shadow:none}.grid{grid-template-columns:repeat(2,minmax(0,1fr))}.metrics{grid-template-columns:repeat(4,minmax(0,1fr))}.metrics.three{grid-template-columns:repeat(3,minmax(0,1fr))}.reference-grid{grid-template-columns:repeat(5,minmax(0,1fr))}section{break-inside:auto}.subcard,figure,.metric{break-inside:avoid}.dot,.prob>div,.prob i,header,.reference.selected,.note,.warning{-webkit-print-color-adjust:exact!important;print-color-adjust:exact!important}}
  </style></head><body>
  <div class="toolbar"><button onclick="window.print()">Print / Save as PDF</button><button onclick="window.close()">Close</button></div>
  <main><header><h1>Paediatric Elbow Fracture Analysis Report</h1><p>Automated Gartland classification and supporting assessments</p><div class="meta"><strong>Generated:</strong> ${escapeHtml(generatedAt)}<br><strong>Minimum confidence level:</strong> ${(confidenceThreshold * 100).toFixed(0)}%</div></header>
  <div class="grid input-images">${reportImage(apImage,"AP view")}${reportImage(latImage,"LAT view")}</div>
  <div class="reference-block"><h2>Gartland Classification Reference</h2><div class="reference-grid">${reference}</div><p class="note">AHL = Anterior Humeral Line drawn along the anterior cortex of the humerus on the lateral X-ray.</p></div>
  <section><h2>1. Classification results</h2><div class="metrics">${metric("Gartland grade",result.final_grade ?? "Unable to grade")}${metric("CNN",result.cnn_grade ?? "n/a")}${metric("Bone geometry",result.geometric_grade ?? "n/a")}${metric("Agreement",agreement)}</div></section>
  <section><h2>2. CNN experiment results</h2><div class="grid">${experimentHtml("exp1",result.experiments.exp1,result)}${experimentHtml("exp2",result.experiments.exp2,result)}${experimentHtml("exp3",result.experiments.exp3,result)}${experimentHtml("exp4",result.experiments.exp4,result)}</div></section>
  <section><h2>3. Baumann angle results</h2>${baumannUnavailable
    ? `<p class="unavailable">${escapeHtml(baumannUnavailable)}</p>`
    : `<div class="metrics three">${metric("Baumann angle",b?.baumann_angle_deg != null?`${b.baumann_angle_deg.toFixed(1)}°`:"n/a")}${metric("Normal range",b?.in_normal_range==null?"n/a":b.in_normal_range?"Yes (60–84°)":"No")}${metric("Status",b?.status??"Not run")}</div>${reportImage(plotUrl(result,"baumann"),"Baumann angle measurement")}`}</section>
  <section><h2>4. Anterior Humeral Line (AHL) results</h2>${ahlUnavailable
    ? `<p class="unavailable">${escapeHtml(ahlUnavailable)}</p>`
    : `<div class="metrics">${metric("Grade 1 vs 2",g?.grade_1v2??"n/a")}${metric("Right of AHL",ahl.split_pct_pos!=null?`${ahl.split_pct_pos.toFixed(1)}%`:"n/a")}${metric("Left of AHL",ahl.split_pct_neg!=null?`${ahl.split_pct_neg.toFixed(1)}%`:"n/a")}${metric("Bisection quality",ahl.bisection_quality_pct!=null?`${ahl.bisection_quality_pct.toFixed(1)}%`:"n/a")}</div>${reportImage(plotUrl(result,"geometric"),"AHL and SAM2 segmentation overlay")}`}</section>
  <section><h2>5. Bone profile results</h2>${boneProfileUnavailable
    ? `<p class="unavailable">${escapeHtml(boneProfileUnavailable)}</p>`
    : `<div class="metrics three">${metric("Predicted sub-grade",g?.grade_2ab??"n/a")}${metric("Match ratio",matchRatio?.toFixed(2)??"n/a")}${metric("Uniformity",uniformity)}</div>${reportImage(plotUrl(result,"cortical_width"),"Cortical width profile")}`}</section>
  <p class="disclaimer"><strong>Research prototype — not for clinical use.</strong> Results must be interpreted by a qualified clinician and do not replace radiological review.</p>
  </main></body></html>`;
}

export function ReportGenerator({ result, apFile, latFile, confidenceThreshold }: ReportGeneratorProps) {
  const [generating, setGenerating] = useState(false);

  async function generateReport() {
    const reportWindow = window.open("", "_blank");
    if (!reportWindow) {
      window.alert("Please allow pop-ups for this site to generate the report.");
      return;
    }

    setGenerating(true);
    reportWindow.document.write("<p style='font-family:Arial;padding:24px'>Preparing report…</p>");
    try {
      const [apImage, latImage] = await Promise.all([fileToDataUrl(apFile), fileToDataUrl(latFile)]);
      reportWindow.document.open();
      reportWindow.document.write(buildReportHtml(result, apImage, latImage, confidenceThreshold));
      reportWindow.document.close();
    } catch {
      reportWindow.close();
      window.alert("The report could not be generated. Please try again.");
    } finally {
      setGenerating(false);
    }
  }

  return (
    <Button type="button" variant="outline" onClick={generateReport} disabled={generating} className="gap-2">
      {generating ? <Loader2 className="h-4 w-4 animate-spin" /> : <Download className="h-4 w-4" />}
      {generating ? "Generating report…" : "Generate report"}
    </Button>
  );
}
