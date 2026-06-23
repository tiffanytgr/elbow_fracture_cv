"use client";

import { useState } from "react";
import { Loader2, Play } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Sidebar } from "@/components/Sidebar";
import { FileUploader } from "@/components/FileUploader";
import { ResultsBanner } from "@/components/ResultsBanner";
import { SummaryTab } from "@/components/tabs/SummaryTab";
import { PerExperimentTab } from "@/components/tabs/PerExperimentTab";
import { BaumannTab } from "@/components/tabs/BaumannTab";
import { GeometricTab } from "@/components/tabs/GeometricTab";
import { CorticalWidthTab } from "@/components/tabs/CorticalWidthTab";
import { AuditTab } from "@/components/tabs/AuditTab";
import type { GraderConfig, PredictResponse } from "@/lib/types";

const DEFAULT_CONFIG: GraderConfig = {
  confidenceThreshold: 0.7,
  runFullLatAlignment: true,
  runSam2: true,
};

function StepBadge({ n, done }: { n: number; done: boolean }) {
  return (
    <span
      className={`inline-flex items-center justify-center w-6 h-6 rounded-full text-xs font-bold flex-shrink-0
        ${done ? "bg-green-500 text-white" : "bg-slate-200 text-slate-600"}`}
    >
      {done ? "✓" : n}
    </span>
  );
}

export default function HomePage() {
  const [apFile, setApFile] = useState<File | null>(null);
  const [latFile, setLatFile] = useState<File | null>(null);
  const [config, setConfig] = useState<GraderConfig>(DEFAULT_CONFIG);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const canRun = apFile !== null || latFile !== null;

  async function handleRun() {
    if (!canRun) return;
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const form = new FormData();
      if (apFile) form.append("ap_file", apFile);
      if (latFile) form.append("lat_file", latFile);
      form.append("confidence_threshold", String(config.confidenceThreshold));
      form.append("run_full_lat_alignment", String(config.runFullLatAlignment));
      form.append("run_sam2", String(config.runSam2));

      const res = await fetch("/api/predict", { method: "POST", body: form });
      const data = await res.json();

      if (!res.ok) {
        setError(data.detail ?? data.error ?? `Error ${res.status}`);
        return;
      }

      setResult(data as PredictResponse);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex min-h-screen">
      {/* Sidebar */}
      <Sidebar
        modelStatus={result?.model_status ?? null}
        config={config}
        onConfigChange={setConfig}
        device={
          result?.config_snapshot
            ? String((result.config_snapshot as Record<string, unknown>)["device"] ?? "")
            : undefined
        }
      />

      {/* Main content */}
      <main className="flex-1 p-6 max-w-6xl space-y-6">
        {/* Header */}
        <div className="rounded-xl bg-gradient-to-r from-[#1e3a5f] to-[#2563a8] px-7 py-6 text-white">
          <h1 className="text-2xl font-bold mb-1">Paediatric Elbow Fracture Grader</h1>
          <p className="text-white/80 text-sm">
            Automated Gartland classification for supracondylar humerus fractures · KKH AI Research
          </p>
        </div>

        {/* Disclaimer */}
        <div className="flex items-start gap-3 rounded-lg border border-orange-200 bg-orange-50 px-4 py-3 text-sm text-orange-900">
          <span className="flex-shrink-0 font-bold">⚠️</span>
          <span>
            <strong>Research prototype — not for clinical use.</strong>{" "}
            Results must be interpreted by a qualified clinician and do not replace radiological review.
          </span>
        </div>

        {/* Step 1 — Upload */}
        <section className="space-y-3">
          <div className="flex items-center gap-2">
            <StepBadge n={1} done={canRun} />
            <h2 className="text-base font-semibold">Upload X-ray Images</h2>
          </div>
          <p className="text-sm text-muted-foreground pl-8">
            Upload the <strong>AP (anteroposterior) view</strong> to screen for fractures.
            Add the <strong>LAT (lateral) view</strong> to enable full Gartland sub-grading
            (Grade 1 / 2a / 2b / 3).
          </p>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <FileUploader
              label="AP View — Anteroposterior"
              hint="Required for fracture screening"
              file={apFile}
              onFileChange={setApFile}
            />
            <FileUploader
              label="LAT View — Lateral (optional)"
              hint="Required for Grade 1 vs 2 sub-grading"
              file={latFile}
              onFileChange={setLatFile}
            />
          </div>
        </section>

        {/* Step 2 — Analyse */}
        <section>
          <div className="flex items-center gap-2 mb-3">
            <StepBadge n={2} done={result !== null} />
            <h2 className="text-base font-semibold">Run Analysis</h2>
          </div>

          <div className="flex items-center gap-4 pl-8">
            <Button
              size="lg"
              onClick={handleRun}
              disabled={!canRun || loading}
              className="gap-2 bg-gradient-to-r from-[#1e3a5f] to-[#2563a8] hover:from-[#1e3a5f]/90 hover:to-[#2563a8]/90"
            >
              {loading ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Play className="w-4 h-4" />
              )}
              {loading ? "Analysing…" : "Analyse X-rays"}
            </Button>
            {!canRun && (
              <p className="text-sm text-muted-foreground">
                Upload at least one X-ray to enable analysis.
              </p>
            )}
            {loading && (
              <p className="text-sm text-muted-foreground animate-pulse">
                First run loads AI models — allow 30–60 s…
              </p>
            )}
          </div>
        </section>

        {/* Error */}
        {error && (
          <div className="rounded-lg bg-red-50 border border-red-200 px-4 py-3 text-sm text-red-800">
            <strong>Error:</strong> {error}
          </div>
        )}

        {/* Step 3 — Results */}
        {result && (
          <section className="space-y-4 border-t border-border pt-6">
            <div className="flex items-center gap-2">
              <StepBadge n={3} done={true} />
              <h2 className="text-base font-semibold">Review Results</h2>
            </div>

            <div className="pl-8 space-y-4">
              <ResultsBanner result={result} hasLat={latFile !== null} />

              <Tabs defaultValue="summary" className="w-full">
                <TabsList className="flex flex-wrap h-auto gap-1 w-full justify-start">
                  <TabsTrigger value="summary">Summary</TabsTrigger>
                  <TabsTrigger value="experiments">CNN Classification</TabsTrigger>
                  <TabsTrigger value="baumann">Baumann Angle (AP)</TabsTrigger>
                  <TabsTrigger value="geometric">AHL Measurement (LAT)</TabsTrigger>
                  <TabsTrigger value="cortical">Bone Profile</TabsTrigger>
                  <TabsTrigger value="audit">Audit Trail</TabsTrigger>
                </TabsList>

                <TabsContent value="summary">
                  <SummaryTab result={result} />
                </TabsContent>
                <TabsContent value="experiments">
                  <PerExperimentTab result={result} />
                </TabsContent>
                <TabsContent value="baumann">
                  <BaumannTab result={result} />
                </TabsContent>
                <TabsContent value="geometric">
                  <GeometricTab result={result} />
                </TabsContent>
                <TabsContent value="cortical">
                  <CorticalWidthTab result={result} />
                </TabsContent>
                <TabsContent value="audit">
                  <AuditTab result={result} />
                </TabsContent>
              </Tabs>
            </div>
          </section>
        )}
      </main>
    </div>
  );
}
