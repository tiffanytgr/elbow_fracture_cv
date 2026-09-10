"use client";

import { useState } from "react";
import { ClipboardList, Grid2X2, Images, Sparkles, UploadCloud } from "lucide-react";

import { FileUploader } from "@/components/FileUploader";
import {
  DEMO_CASES,
  DemoCaseSelector,
  type DemoCase,
} from "@/components/DemoCaseSelector";
import { CaseImageBoard } from "@/components/CaseImageBoard";
import { AssessmentPanel } from "@/components/AssessmentPanel";

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
  const [uploadVersion, setUploadVersion] = useState(0);
  const [demoVersion, setDemoVersion] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const apFile = inputMode === "upload" ? uploadedApFile : demoApFile;
  const latFile = inputMode === "upload" ? uploadedLatFile : demoLatFile;
  const hasCase = apFile !== null || latFile !== null;

  const caseId =
    inputMode === "demo"
      ? selectedDemoId
      : [uploadedApFile?.name, uploadedLatFile?.name]
          .filter(Boolean)
          .join(" + ") || null;
  const caseKey = !hasCase
    ? null
    : inputMode === "demo"
      ? `demo:${selectedDemoId}:${demoVersion}`
      : `upload:${uploadVersion}`;

  function changeInputMode(mode: "upload" | "demo") {
    if (loadingDemoId !== null || mode === inputMode) return;
    setInputMode(mode);
    setError(null);
  }

  async function handleDemoSelect(demo: DemoCase) {
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

  return (
    <div className="min-h-screen bg-[#f8fafc]">
      <main className="mx-auto min-w-0 max-w-5xl space-y-4 p-4 sm:p-6">
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
              Paediatric Elbow Fracture — Clinician Grading
            </h1>
            <p className="mt-2 text-sm font-medium text-white/90 sm:text-base">
              Manual Gartland classification of supracondylar humerus fractures
            </p>
            <div className="mt-4">
              <span className="inline-flex items-center gap-1.5 rounded-full border border-blue-300/20 bg-blue-500/60 px-3 py-1.5 text-xs font-semibold text-white shadow-sm backdrop-blur-sm">
                <ClipboardList className="h-4 w-4" />
                Control study — no AI assistance
              </span>
            </div>
          </div>
        </div>

        {/* Step 1 — Choose images */}
        <section className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
          <div className="flex items-center gap-2">
            <StepBadge n={1} done={hasCase} />
            <h2 className="text-base font-semibold">Choose X-ray Images</h2>
          </div>
          <p className="mt-1 pl-8 text-sm text-muted-foreground">
            Upload your own AP/LAT images, or select an example case to grade.
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
                disabled={loadingDemoId !== null}
                aria-pressed={inputMode === "demo"}
                className={`flex items-center justify-center gap-2 rounded-lg border px-4 py-3 text-sm font-semibold transition disabled:cursor-not-allowed disabled:opacity-50 ${
                  inputMode === "demo"
                    ? "border-blue-600 bg-blue-600 text-white shadow-sm"
                    : "border-slate-300 bg-white text-slate-700 hover:border-blue-300"
                }`}
              >
                <Images className="h-4 w-4" />
                Example cases
              </button>
              <button
                type="button"
                onClick={() => changeInputMode("upload")}
                disabled={loadingDemoId !== null}
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
                  file={uploadedApFile}
                  onFileChange={(file) => {
                    setUploadedApFile(file);
                    setUploadVersion((version) => version + 1);
                  }}
                />
                <FileUploader
                  label="LAT View — Lateral (optional)"
                  file={uploadedLatFile}
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
                        Select an example case
                      </p>
                      <p className="text-xs text-slate-500">
                        Grade each case from the X-rays alone.
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
                  onSelect={handleDemoSelect}
                />
              </div>
            )}
          </div>
        </section>

        {error && (
          <div className="rounded-lg bg-red-50 border border-red-200 px-4 py-3 text-sm text-red-800">
            <strong>Error:</strong> {error}
          </div>
        )}

        {/* Step 2 — Review & grade */}
        {hasCase && (
          <section className="space-y-4 rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
            <div className="flex items-center gap-2">
              <StepBadge n={2} done={false} />
              <h2 className="text-base font-semibold">Review &amp; Grade</h2>
            </div>
            <div className="pl-8">
              <CaseImageBoard apFile={apFile} latFile={latFile} />
            </div>
          </section>
        )}

        {hasCase && (
          <AssessmentPanel
            caseKey={caseKey}
            caseId={caseId}
            inputMode={inputMode}
          />
        )}
      </main>
    </div>
  );
}
