"use client";

import { useCallback, useEffect, useState } from "react";
import {
  FlaskConical,
  Grid2X2,
  Images,
  Loader2,
  Play,
  ShieldCheck,
  Sparkles,
  UploadCloud,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Sidebar } from "@/components/Sidebar";
import { StudyPanel } from "@/components/StudyPanel";
import { CaseImageBoard } from "@/components/CaseImageBoard";
import {
  FileUploader,
  filePathOf,
  type FileChangeKind,
} from "@/components/FileUploader";
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
import type { StudyMode } from "@/lib/studyTypes";

const DEFAULT_CONFIG: GraderConfig = {
  confidenceThreshold: 0.7,
  runFullLatAlignment: true,
  runSam2: true,
};

const REVIEWER_STORAGE_KEY = "elbow-grader-reviewer";
const MODE_STORAGE_KEY = "elbow-grader-mode";

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
  const [uploadedApPath, setUploadedApPath] = useState<string | null>(null);
  const [uploadedLatPath, setUploadedLatPath] = useState<string | null>(null);
  // Bumped whenever uploaded images start a new case (see handleUpload).
  const [uploadCaseSeq, setUploadCaseSeq] = useState(0);
  const [caseIdInput, setCaseIdInput] = useState("");
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

  // Study-session state.
  const [reviewer, setReviewer] = useState("");
  const [mode, setMode] = useState<StudyMode>("ai");
  const [preLocked, setPreLocked] = useState(false);
  const [savedCaseKey, setSavedCaseKey] = useState<string | null>(null);

  const apFile = inputMode === "upload" ? uploadedApFile : demoApFile;
  const latFile = inputMode === "upload" ? uploadedLatFile : demoLatFile;
  const canRun = apFile !== null || latFile !== null;
  const currentInputKey = `${inputMode}:${
    inputMode === "upload" ? uploadVersion : demoVersion
  }:${configVersion}`;
  const resultIsStale = result !== null && currentInputKey !== resultInputKey;

  // Source path of each loaded X-ray; the case identifier is built from these.
  const selectedDemo = DEMO_CASES.find((d) => d.id === selectedDemoId) ?? null;
  const apPath =
    inputMode === "upload"
      ? uploadedApPath
      : demoApFile
        ? selectedDemo?.apUrl ?? null
        : null;
  const latPath =
    inputMode === "upload"
      ? uploadedLatPath
      : demoLatFile
        ? selectedDemo?.latUrl ?? null
        : null;
  // Case identifier entered by the reader; it is what the logs are keyed on.
  const caseId = caseIdInput.trim() || null;
  // Identity of the loaded case, used to drive the timer and per-case reset.
  const caseKey = !canRun
    ? null
    : inputMode === "demo"
      ? `demo:${selectedDemoId}:${demoVersion}`
      : `upload:${uploadCaseSeq}`;

  const isControl = mode === "control";
  // A case is "in progress" once loaded and until it has been saved; the study
  // arm is locked during this window so it can't be flipped mid-case.
  const caseInProgress = caseKey !== null && caseKey !== savedCaseKey;
  const aiRevealed = !isControl && result !== null && !resultIsStale;

  // Restore reviewer + arm from a previous session.
  useEffect(() => {
    try {
      const storedReviewer = window.localStorage.getItem(REVIEWER_STORAGE_KEY);
      if (storedReviewer) setReviewer(storedReviewer);
      const storedMode = window.localStorage.getItem(MODE_STORAGE_KEY);
      if (storedMode === "ai" || storedMode === "control") setMode(storedMode);
    } catch {
      /* localStorage may be unavailable */
    }
  }, []);

  // Clear any AI result whenever a new case loads or the arm changes, so the
  // post-AI read always waits for a fresh analysis of the current case.
  useEffect(() => {
    setResult(null);
    setError(null);
  }, [caseKey, mode]);

  function persistReviewer(next: string) {
    setReviewer(next);
    try {
      window.localStorage.setItem(REVIEWER_STORAGE_KEY, next.trim());
    } catch {
      /* ignore */
    }
  }

  function changeMode(next: StudyMode) {
    if (caseInProgress || next === mode) return;
    setMode(next);
    try {
      window.localStorage.setItem(MODE_STORAGE_KEY, next);
    } catch {
      /* ignore */
    }
  }

  /**
   * Uploading new X-ray images starts a new case. The one exception is adding
   * the missing second view to an unsaved case (e.g. LAT after AP), which
   * completes the current case. Rotating or removing an image never does.
   */
  function handleUpload(view: "ap" | "lat", file: File | null, kind: FileChangeKind) {
    const slotFile = view === "ap" ? uploadedApFile : uploadedLatFile;
    const otherFile = view === "ap" ? uploadedLatFile : uploadedApFile;
    const setFile = view === "ap" ? setUploadedApFile : setUploadedLatFile;
    const setPath = view === "ap" ? setUploadedApPath : setUploadedLatPath;

    setFile(file);
    if (kind !== "rotate") setPath(file ? filePathOf(file) : null);
    setUploadVersion((version) => version + 1);

    if (kind === "new") {
      const currentKey = `upload:${uploadCaseSeq}`;
      const completesOpenCase =
        slotFile === null && otherFile !== null && savedCaseKey !== currentKey;
      if (!completesOpenCase) setUploadCaseSeq((seq) => seq + 1);
    }
  }

  const handleSaved = useCallback((key: string) => {
    setSavedCaseKey(key);
    // Clear the ID so the next case can't be logged under this one by mistake.
    setCaseIdInput("");
  }, []);

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
      setCaseIdInput(demo.id);
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
    if (!canRun || loadingDemoId !== null || isControl || !preLocked) return;
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
      if (apFile && apPath) form.append("ap_source_path", apPath);
      if (latFile && latPath) form.append("lat_source_path", latPath);
      if (caseId) form.append("case_id", caseId);

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
      {/* Sidebar — hidden in the control arm so no AI settings/telemetry show */}
      {!isControl && (
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
      )}

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
              Gartland classification reader study — AI-assisted and control arms
            </p>
            <div className="mt-4">
              <span className="inline-flex items-center gap-1.5 rounded-full border border-blue-300/20 bg-blue-500/60 px-3 py-1.5 text-xs font-semibold text-white shadow-sm backdrop-blur-sm">
                <ShieldCheck className="h-4 w-4" />
                KKH AI Research
              </span>
            </div>
          </div>
        </div>

        {/* Study session — reviewer + arm, locked once a case is in progress */}
        <section className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
          <div className="flex items-center gap-2">
            <FlaskConical className="h-5 w-5 text-blue-600" />
            <h2 className="text-base font-semibold">Study Session</h2>
          </div>
          <div className="mt-4 grid grid-cols-1 gap-4 md:grid-cols-2">
            <label className="text-sm">
              <span className="mb-1 block font-medium text-slate-700">
                Reviewer
              </span>
              <input
                type="text"
                value={reviewer}
                onChange={(e) => persistReviewer(e.target.value)}
                placeholder="Your name or initials"
                className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
              />
            </label>
            <div className="text-sm">
              <span className="mb-1 block font-medium text-slate-700">
                Study arm
              </span>
              <div
                className="grid grid-cols-2 gap-2"
                role="group"
                aria-label="Study arm"
              >
                <button
                  type="button"
                  onClick={() => changeMode("ai")}
                  disabled={caseInProgress}
                  aria-pressed={mode === "ai"}
                  className={`rounded-lg border px-3 py-2 text-sm font-semibold transition disabled:cursor-not-allowed disabled:opacity-50 ${
                    mode === "ai"
                      ? "border-blue-600 bg-blue-600 text-white shadow-sm"
                      : "border-slate-300 bg-white text-slate-700 hover:border-blue-300"
                  }`}
                >
                  AI-assisted
                </button>
                <button
                  type="button"
                  onClick={() => changeMode("control")}
                  disabled={caseInProgress}
                  aria-pressed={mode === "control"}
                  className={`rounded-lg border px-3 py-2 text-sm font-semibold transition disabled:cursor-not-allowed disabled:opacity-50 ${
                    mode === "control"
                      ? "border-blue-600 bg-blue-600 text-white shadow-sm"
                      : "border-slate-300 bg-white text-slate-700 hover:border-blue-300"
                  }`}
                >
                  Control (no AI)
                </button>
              </div>
              <p className="mt-1 text-xs text-slate-500">
                {caseInProgress
                  ? "Arm is locked while a case is in progress. Submit the assessment to change it."
                  : isControl
                    ? "Control arm: images only, no AI output. One read per case."
                    : "AI-assisted arm: record your read before and after seeing the AI."}
              </p>
            </div>
          </div>
        </section>

        {/* Step 1 — Choose images */}
        <section className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
          <div className="flex items-center gap-2">
            <StepBadge n={1} done={canRun && caseId !== null} />
            <h2 className="text-base font-semibold">Enter Case ID and Choose X-ray Images</h2>
          </div>
          <p className="mt-1 pl-8 text-sm text-muted-foreground">
            Enter the case ID, then upload your own AP/LAT images or select a
            Grade 2a or Grade 2b example case.
          </p>

          <div className="pl-8">
            <label className="mt-4 block w-full max-w-xl text-sm">
              <span className="mb-1 block font-medium text-slate-700">
                Case ID <span className="text-red-600">*</span>
              </span>
              <input
                type="text"
                value={caseIdInput}
                onChange={(e) => setCaseIdInput(e.target.value)}
                placeholder="e.g. KKH-0123"
                aria-required="true"
                className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
              />
              <span className="mt-1 block text-xs text-slate-500">
                Used as the case identifier in the logs. Required before grading.
              </span>
            </label>
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
                Example case
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
                  onFileChange={(file, kind) => handleUpload("ap", file, kind)}
                />
                <FileUploader
                  label="LAT View — Lateral (optional)"
                  hint="Required for Grade 1 vs 2 sub-grading"
                  file={uploadedLatFile}
                  disabled={loading}
                  onFileChange={(file, kind) => handleUpload("lat", file, kind)}
                />
              </div>
            ) : (
              <div className="mt-4 space-y-3">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div className="flex items-start gap-3">
                    <Sparkles className="mt-0.5 h-5 w-5 text-blue-600" />
                    <div>
                    <p className="text-xs font-bold uppercase tracking-wider text-slate-700">
                      Select an example case
                    </p>
                    <p className="text-xs text-slate-500">
                      Curated Grade 2a / 2b studies.
                    </p>
                    </div>
                  </div>
                  <span className="inline-flex items-center gap-2 rounded-full bg-blue-50 px-3 py-1.5 text-xs font-medium text-blue-700">
                    <Grid2X2 className="h-3.5 w-3.5" />
                    {DEMO_CASES.length} cases
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

        {/* Review images — shown for the loaded case (both arms) */}
        {canRun && (
          <section className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
            <div className="flex items-center gap-2">
              <Images className="h-5 w-5 text-blue-600" />
              <h2 className="text-base font-semibold">Review Images</h2>
            </div>
            <p className="mt-1 text-sm text-muted-foreground">
              Click either view to enlarge.
            </p>
            <div className="mt-4">
              <CaseImageBoard apFile={apFile} latFile={latFile} />
            </div>
          </section>
        )}

        {/* Reader assessment — timer + pre/post reads + save */}
        <StudyPanel
          caseKey={caseKey}
          caseId={caseId}
          apPath={apPath}
          latPath={latPath}
          inputMode={inputMode}
          reviewer={reviewer}
          mode={mode}
          aiRevealed={aiRevealed}
          aiGartlandGrade={aiRevealed ? result?.final_grade ?? null : null}
          aiCnnGrade={aiRevealed ? result?.cnn_grade ?? null : null}
          aiGeometricGrade={aiRevealed ? result?.geometric_grade ?? null : null}
          aiConfidence={aiRevealed ? result?.confidence ?? null : null}
          onPreLockedChange={setPreLocked}
          onSaved={handleSaved}
        />

        {/* Step 2 — Analyse (AI arm only) */}
        {!isControl && (
          <section className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
            <div className="flex items-center gap-2 mb-3">
              <StepBadge n={2} done={result !== null} />
              <h2 className="text-base font-semibold">Run AI Analysis</h2>
            </div>

            <div className="flex flex-wrap items-center gap-4 pl-8">
              <Button
                size="lg"
                onClick={handleRun}
                disabled={
                  !canRun || !caseId || loading || loadingDemoId !== null || !preLocked
                }
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
                  {inputMode === "demo"
                    ? "Select a case to enable analysis."
                    : "Upload at least one X-ray to enable analysis."}
                </p>
              )}
              {canRun && !preLocked && (
                <p className="text-sm text-amber-700">
                  Lock your pre-AI read above before running the analysis.
                </p>
              )}
              {loading && (
                <p className="text-sm text-muted-foreground animate-pulse">
                  First run loads AI models — allow 30–60 s…
                </p>
              )}
            </div>
          </section>
        )}

        {/* Error */}
        {error && (
          <div className="rounded-lg bg-red-50 border border-red-200 px-4 py-3 text-sm text-red-800">
            <strong>Error:</strong> {error}
          </div>
        )}

        {/* Step 3 — Results (AI arm only) */}
        {!isControl && result && (
          <section className="space-y-4 rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div className="flex items-center gap-2">
                <StepBadge n={3} done={true} />
                <h2 className="text-base font-semibold">Review AI Results</h2>
              </div>
              <ReportGenerator
                result={result}
                apFile={resultApFile}
                latFile={resultLatFile}
                confidenceThreshold={resultConfidenceThreshold}
              />
            </div>
            <p className="pl-8 text-sm text-muted-foreground">
              AI analysis complete. Review it, then choose and submit your grade in the
              assessment panel above.
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
