"use client";

import { useState } from "react";
import { Grid2X2, Images, Loader2, Play, ShieldCheck, Sparkles, UploadCloud } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Sidebar } from "@/components/Sidebar";
import { FileUploader } from "@/components/FileUploader";
import {
  DEMO_CASES,
  DemoCaseSelector,
  type DemoCase,
} from "@/components/DemoCaseSelector";
import { GartlandReference, ResultsBanner } from "@/components/ResultsBanner";
import { SummaryTab } from "@/components/tabs/SummaryTab";
import { PerExperimentTab } from "@/components/tabs/PerExperimentTab";
import { BaumannTab } from "@/components/tabs/BaumannTab";
import { GeometricTab } from "@/components/tabs/GeometricTab";
import { CorticalWidthTab } from "@/components/tabs/CorticalWidthTab";
import { ReportGenerator } from "@/components/ReportGenerator";
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
  const [inputMode, setInputMode] = useState<"upload" | "demo">("demo");
  const [uploadedApFile, setUploadedApFile] = useState<File | null>(null);
  const [uploadedLatFile, setUploadedLatFile] = useState<File | null>(null);
  const [demoApFile, setDemoApFile] = useState<File | null>(null);
  const [demoLatFile, setDemoLatFile] = useState<File | null>(null);
  const [selectedDemoId, setSelectedDemoId] = useState<string | null>(null);
  const [loadingDemoId, setLoadingDemoId] = useState<string | null>(null);
  const [config, setConfig] = useState<GraderConfig>(DEFAULT_CONFIG);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [uploadVersion, setUploadVersion] = useState(0);
  const [demoVersion, setDemoVersion] = useState(0);
  const [configVersion, setConfigVersion] = useState(0);
  const [resultInputKey, setResultInputKey] = useState<string | null>(null);
  const [resultHasLat, setResultHasLat] = useState(false);
  const [resultApFile, setResultApFile] = useState<File | null>(null);
  const [resultLatFile, setResultLatFile] = useState<File | null>(null);
  const [resultConfidenceThreshold, setResultConfidenceThreshold] = useState(DEFAULT_CONFIG.confidenceThreshold);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const apFile = inputMode === "upload" ? uploadedApFile : demoApFile;
  const latFile = inputMode === "upload" ? uploadedLatFile : demoLatFile;
  const canRun = apFile !== null || latFile !== null;
  const currentInputKey = `${inputMode}:${
    inputMode === "upload" ? uploadVersion : demoVersion
  }:${configVersion}`;
  const resultIsStale = result !== null && currentInputKey !== resultInputKey;

  function changeInputMode(mode: "upload" | "demo") {
    if (loading || loadingDemoId !== null || mode === inputMode) return;
    setInputMode(mode);
    setError(null);
  }

  async function handleDemoSelect(demo: DemoCase) {
    if (loading) return;
    setLoadingDemoId(demo.id);
    setError(null);

    try {
      const [latResponse, apResponse] = await Promise.all([
        fetch(demo.latUrl),
        demo.apUrl ? fetch(demo.apUrl) : Promise.resolve(null),
      ]);

      if (!latResponse.ok || (apResponse && !apResponse.ok)) {
        throw new Error("The selected demo images could not be loaded.");
      }

      const [latBlob, apBlob] = await Promise.all([
        latResponse.blob(),
        apResponse ? apResponse.blob() : Promise.resolve(null),
      ]);

      setDemoApFile(
        apBlob
          ? new File([apBlob], `${demo.id}-ap.png`, {
              type: apBlob.type || "image/png",
            })
          : null,
      );
      setDemoLatFile(
        new File([latBlob], `${demo.id}-lat.png`, {
          type: latBlob.type || "image/png",
        }),
      );
      setSelectedDemoId(demo.id);
      setDemoVersion((version) => version + 1);
    } catch (e) {
      setDemoApFile(null);
      setDemoLatFile(null);
      setSelectedDemoId(null);
      setDemoVersion((version) => version + 1);
      setError(e instanceof Error ? e.message : "Unable to load the demo case.");
    } finally {
      setLoadingDemoId(null);
    }
  }

  async function handleRun() {
    if (!canRun || loadingDemoId !== null) return;
    const submittedInputKey = currentInputKey;
    const submittedHasLat = latFile !== null;
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
      setResultInputKey(submittedInputKey);
      setResultHasLat(submittedHasLat);
      setResultApFile(apFile);
      setResultLatFile(latFile);
      setResultConfidenceThreshold(config.confidenceThreshold);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-[#f8fafc] lg:flex">
      {/* Sidebar */}
      <Sidebar
        modelStatus={result?.model_status ?? null}
        config={config}
        result={result}
        onConfigChange={(nextConfig) => {
          setConfig(nextConfig);
          setConfigVersion((version) => version + 1);
        }}
        device={
          result?.config_snapshot
            ? String((result.config_snapshot as Record<string, unknown>)["device"] ?? "")
            : undefined
        }
      />

      {/* Main content */}
      <main className="min-w-0 flex-1 space-y-4 p-4 sm:p-6 lg:p-5 xl:p-6">
        {/* Header */}
        <div
          className="relative isolate min-h-[180px] overflow-hidden rounded-2xl bg-[#06245c] px-7 py-7 text-white shadow-sm sm:px-9 sm:py-8"
          style={{
            backgroundImage: "url('/elbow-header-background.png')",
            backgroundPosition: "center right",
            backgroundSize: "cover",
          }}
        >
          <div className="absolute inset-0 -z-10 bg-gradient-to-r from-[#073272]/95 via-[#06275f]/80 to-[#033b87]/10" />
          <div className="relative flex min-h-[124px] max-w-3xl flex-col justify-center">
            <h1 className="text-3xl font-bold tracking-tight sm:text-[2.5rem]">
              Paediatric Elbow Fracture Grader
            </h1>
            <p className="mt-2 text-sm font-medium text-white/90 sm:text-base">
              Automated Gartland classification for supracondylar humerus fractures
            </p>
            <div className="mt-4">
              <span className="inline-flex items-center gap-1.5 rounded-full border border-blue-300/20 bg-blue-500/60 px-3 py-1.5 text-xs font-semibold text-white shadow-sm backdrop-blur-sm">
                <ShieldCheck className="h-4 w-4" />
                KKH AI Research
              </span>
            </div>
          </div>
        </div>

        {/* Step 1 — Choose images */}
        <section className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
          <div className="flex items-center gap-2">
            <StepBadge n={1} done={canRun} />
            <h2 className="text-base font-semibold">Choose X-ray Images</h2>
          </div>
          <p className="mt-1 pl-8 text-sm text-muted-foreground">
            Upload your own AP/LAT images, or select a Grade 2a or Grade 2b
            example to explore the analysis workflow.
          </p>

          <div className="pl-8">
            <div
              className="mt-4 grid w-full max-w-xl grid-cols-2 gap-3"
              role="group"
              aria-label="Choose image source"
            >
              <button
                type="button"
                onClick={() => changeInputMode("demo")}
                disabled={loading || loadingDemoId !== null}
                aria-pressed={inputMode === "demo"}
                className={`flex items-center justify-center gap-2 rounded-lg border px-4 py-3 text-sm font-semibold transition disabled:cursor-not-allowed disabled:opacity-50 ${
                  inputMode === "demo"
                    ? "border-blue-600 bg-blue-600 text-white shadow-sm"
                    : "border-slate-300 bg-white text-slate-700 hover:border-blue-300"
                }`}
              >
                <Images className="h-4 w-4" />
                Try a demo case
              </button>
              <button
                type="button"
                onClick={() => changeInputMode("upload")}
                disabled={loading || loadingDemoId !== null}
                aria-pressed={inputMode === "upload"}
                className={`flex items-center justify-center gap-2 rounded-lg border px-4 py-3 text-sm font-semibold transition disabled:cursor-not-allowed disabled:opacity-50 ${
                  inputMode === "upload"
                    ? "border-blue-600 bg-blue-600 text-white shadow-sm"
                    : "border-slate-300 bg-white text-slate-700 hover:border-blue-300"
                }`}
              >
                <UploadCloud className="h-4 w-4" />
                Upload your own
              </button>
            </div>

            {inputMode === "upload" ? (
              <div className="mt-6 grid grid-cols-1 gap-6 md:grid-cols-2">
                <FileUploader
                  label="AP View — Anteroposterior"
                  hint="Required for fracture screening"
                  file={uploadedApFile}
                  disabled={loading}
                  onFileChange={(file) => {
                    setUploadedApFile(file);
                    setUploadVersion((version) => version + 1);
                  }}
                />
                <FileUploader
                  label="LAT View — Lateral (optional)"
                  hint="Required for Grade 1 vs 2 sub-grading"
                  file={uploadedLatFile}
                  disabled={loading}
                  onFileChange={(file) => {
                    setUploadedLatFile(file);
                    setUploadVersion((version) => version + 1);
                  }}
                />
              </div>
            ) : (
              <div className="mt-4 space-y-3">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div className="flex items-start gap-3">
                    <Sparkles className="mt-0.5 h-5 w-5 text-blue-600" />
                    <div>
                    <p className="text-xs font-bold uppercase tracking-wider text-slate-700">
                      Select a demo example
                    </p>
                    <p className="text-xs text-slate-500">
                      Explore curated cases to see how the model performs.
                    </p>
                    </div>
                  </div>
                  <span className="inline-flex items-center gap-2 rounded-full bg-blue-50 px-3 py-1.5 text-xs font-medium text-blue-700">
                    <Grid2X2 className="h-3.5 w-3.5" />
                    {DEMO_CASES.length} demo cases
                  </span>
                </div>
                <DemoCaseSelector
                  selectedId={selectedDemoId}
                  loadingId={loadingDemoId}
                  disabled={loading}
                  onSelect={handleDemoSelect}
                />
              </div>
            )}
          </div>
        </section>

        {/* Step 2 — Analyse */}
        <section className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
          <div className="flex items-center gap-2 mb-3">
            <StepBadge n={2} done={result !== null} />
            <h2 className="text-base font-semibold">Run Analysis</h2>
          </div>

          <div className="flex items-center gap-4 pl-8">
            <Button
              size="lg"
              onClick={handleRun}
              disabled={!canRun || loading || loadingDemoId !== null}
              className="gap-2 bg-gradient-to-r from-[#1e3a5f] to-[#2563a8] hover:from-[#1e3a5f]/90 hover:to-[#2563a8]/90"
            >
              {loading ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Play className="w-4 h-4" />
              )}
              {loading ? "Analysing…" : "Analyse X-rays"}
            </Button>
            {resultIsStale && !loading && (
              <p
                className="text-sm font-medium text-amber-700"
                role="status"
                aria-live="polite"
              >
                New images or settings selected. Results below are from the previous analysis.
              </p>
            )}
            {!canRun && (
              <p className="text-sm text-muted-foreground">
                {inputMode === "demo"
                  ? "Select a demo case to enable analysis."
                  : "Upload at least one X-ray to enable analysis."}
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
          <section className="space-y-4 rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div className="flex items-center gap-2">
                <StepBadge n={3} done={true} />
                <h2 className="text-base font-semibold">Review Results</h2>
              </div>
              <ReportGenerator
                result={result}
                apFile={resultApFile}
                latFile={resultLatFile}
                confidenceThreshold={resultConfidenceThreshold}
              />
            </div>
            <p className="pl-8 text-sm text-muted-foreground">
              AI analysis complete. Please review the classification and supporting assessments.
            </p>

            <div className="pl-8 space-y-4">
              <GartlandReference currentGrade={result.final_grade} />
              <ResultsBanner result={result} hasLat={resultHasLat} />
              <SummaryTab result={result} />

              <div>
              <Tabs defaultValue="experiments" className="w-full">
                <TabsList className="flex h-auto w-full flex-nowrap justify-start gap-1 overflow-x-auto">
                  <TabsTrigger value="experiments">CNN Classification</TabsTrigger>
                  <TabsTrigger value="baumann">Baumann Angle (AP)</TabsTrigger>
                  <TabsTrigger value="geometric">AHL Measurement (LAT)</TabsTrigger>
                  <TabsTrigger value="cortical">Bone Profile</TabsTrigger>
                </TabsList>

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
              </Tabs>
              </div>
            </div>
          </section>
        )}
      </main>
    </div>
  );
}
