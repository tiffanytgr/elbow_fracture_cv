"use client";

import { useEffect, useState } from "react";
import Image from "next/image";
import { Check, Loader2, Maximize2, X } from "lucide-react";

import { cn } from "@/lib/utils";

export interface DemoCase {
  id: string;
  label: string;
  description: string;
  grade: "2a" | "2b";
  apUrl?: string;
  latUrl: string;
}

export const DEMO_CASES: DemoCase[] = [
  {
    id: "a145",
    label: "Demo case A",
    description: "Paired AP and lateral Grade 2a study",
    grade: "2a",
    apUrl: "/demo/grade-2a/a145-ap.png",
    latUrl: "/demo/grade-2a/a145-lat.png",
  },
  {
    id: "a557",
    label: "Demo case B",
    description: "Paired AP and lateral Grade 2a study",
    grade: "2a",
    apUrl: "/demo/grade-2a/a557-ap.png",
    latUrl: "/demo/grade-2a/a557-lat.png",
  },
  {
    id: "a295",
    label: "Demo case C",
    description: "Paired AP and lateral Grade 2a study",
    grade: "2a",
    apUrl: "/demo/grade-2a/a295-ap.png",
    latUrl: "/demo/grade-2a/a295-lat.png",
  },
  {
    id: "a3801",
    label: "Demo case D",
    description: "Paired AP and lateral Grade 2a study",
    grade: "2a",
    apUrl: "/demo/grade-2a/a3801-ap.png",
    latUrl: "/demo/grade-2a/a3801-lat.png",
  },
  {
    id: "a2731",
    label: "Demo case E",
    description: "Lateral-only Grade 2b study",
    grade: "2b",
    latUrl: "/demo/grade-2b/a2731-lat.png",
  },
  {
    id: "a2945",
    label: "Demo case F",
    description: "Lateral-only Grade 2b study",
    grade: "2b",
    latUrl: "/demo/grade-2b/a2945-lat.png",
  },
  {
    id: "a163",
    label: "Demo case G",
    description: "Lateral-only Grade 2b study",
    grade: "2b",
    latUrl: "/demo/grade-2b/a163-lat.png",
  },
  {
    id: "a678",
    label: "Demo case H",
    description: "Lateral-only Grade 2b study",
    grade: "2b",
    latUrl: "/demo/grade-2b/a678-lat.png",
  },
];

interface DemoCaseSelectorProps {
  selectedId: string | null;
  loadingId: string | null;
  disabled?: boolean;
  onSelect: (demo: DemoCase) => void;
}

export function DemoCaseSelector({
  selectedId,
  loadingId,
  disabled = false,
  onSelect,
}: DemoCaseSelectorProps) {
  const [preview, setPreview] = useState<{
    demo: DemoCase;
    view: "AP" | "LAT";
  } | null>(null);

  useEffect(() => {
    if (!preview) return;

    const previousOverflow = document.body.style.overflow;
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setPreview(null);
    };

    document.body.style.overflow = "hidden";
    window.addEventListener("keydown", closeOnEscape);
    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener("keydown", closeOnEscape);
    };
  }, [preview]);

  return (
    <>
      <div className="space-y-5">
        {(["2a", "2b"] as const).map((grade) => (
          <section key={grade} className="space-y-2">
            <div className="flex items-center gap-2">
              <p className={cn(
                "rounded-md px-2 py-1 text-xs font-bold uppercase tracking-wide",
                grade === "2a" ? "bg-blue-50 text-blue-700" : "bg-emerald-50 text-emerald-700",
              )}>
                Grade {grade} demos
              </p>
              <span className="text-xs text-slate-400">
                {grade === "2a" ? "Paired AP + LAT" : "LAT only"}
              </span>
            </div>
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-4">
        {DEMO_CASES.filter((demo) => demo.grade === grade).map((demo) => {
          const selected = selectedId === demo.id;
          const loading = loadingId === demo.id;
          const views: ("AP" | "LAT")[] = demo.apUrl ? ["AP", "LAT"] : ["LAT"];

          return (
            <div
              key={demo.id}
              className={cn(
                "overflow-hidden rounded-xl border bg-white text-left shadow-[0_4px_14px_rgba(15,23,42,0.08)] transition",
                "hover:-translate-y-0.5 hover:border-blue-400 hover:shadow-md",
                selected && "border-blue-600 ring-2 ring-blue-100",
              )}
            >
              <div
                className={cn(
                  "grid h-28 gap-px bg-slate-200 xl:h-32",
                  demo.apUrl ? "grid-cols-2" : "grid-cols-1",
                )}
              >
                {views.map((view) => {
                  const imageUrl = view === "AP" ? demo.apUrl! : demo.latUrl;

                  return (
                    <button
                      key={view}
                      type="button"
                      onClick={() => setPreview({ demo, view })}
                      className="group/image relative bg-slate-950 focus-visible:z-10 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-blue-400"
                      aria-label={`Expand ${demo.label} ${view} X-ray`}
                    >
                      <Image
                        src={imageUrl}
                        alt={`${demo.label} ${view} X-ray`}
                        fill
                        sizes="(max-width: 640px) 50vw, 180px"
                        className="object-contain transition-transform group-hover/image:scale-[1.03]"
                      />
                      <span className="absolute bottom-1 left-1 rounded bg-black/70 px-1.5 py-0.5 text-[10px] font-semibold text-white">
                        {view}
                      </span>
                      <span className="absolute right-1.5 top-1.5 flex h-7 w-7 items-center justify-center rounded-md bg-black/60 text-white opacity-0 transition-opacity group-hover/image:opacity-100 group-focus-visible/image:opacity-100">
                        <Maximize2 className="h-3.5 w-3.5" />
                      </span>
                    </button>
                  );
                })}
              </div>

              <button
                type="button"
                onClick={() => onSelect(demo)}
                disabled={disabled || loadingId !== null}
                aria-pressed={selected}
                className={cn(
                  "flex min-h-20 w-full items-start justify-between gap-3 p-3 text-left",
                  "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-blue-500",
                  "disabled:cursor-not-allowed disabled:opacity-60",
                )}
              >
                <div>
                  <p className="text-sm font-semibold text-slate-900">{demo.label}</p>
                  <p className="mt-0.5 text-xs leading-4 text-slate-500">
                    {demo.description}
                  </p>
                </div>
                <span
                  className={cn(
                    "mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-full",
                    selected
                      ? "bg-blue-600 text-white"
                      : "border border-slate-300 text-transparent",
                  )}
                >
                  {loading ? (
                    <Loader2 className="h-3.5 w-3.5 animate-spin text-blue-600" />
                  ) : (
                    <Check className="h-3.5 w-3.5" />
                  )}
                </span>
              </button>
            </div>
          );
        })}
            </div>
          </section>
        ))}
      </div>

      {preview && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/90 p-3 backdrop-blur-sm sm:p-6"
          onMouseDown={(event) => {
            if (event.target === event.currentTarget) setPreview(null);
          }}
        >
          <div
            role="dialog"
            aria-modal="true"
            aria-labelledby="xray-preview-title"
            className="flex h-[min(92vh,900px)] w-full max-w-7xl flex-col overflow-hidden rounded-xl border border-white/15 bg-slate-950 shadow-2xl"
          >
            <div className="flex flex-wrap items-center justify-between gap-3 border-b border-white/10 px-4 py-3 sm:px-5">
              <div>
                <h3 id="xray-preview-title" className="font-semibold text-white">
                  {preview.demo.label} — {preview.view} view
                </h3>
                <p className="text-xs text-slate-400">
                  Full X-ray preview
                </p>
              </div>

              <div className="flex items-center gap-2">
                <div className="flex rounded-lg bg-white/10 p-1">
                  {(preview.demo.apUrl ? (["AP", "LAT"] as const) : (["LAT"] as const)).map((view) => (
                    <button
                      key={view}
                      type="button"
                      onClick={() => setPreview({ ...preview, view })}
                      aria-pressed={preview.view === view}
                      className={cn(
                        "rounded-md px-3 py-1.5 text-xs font-semibold transition",
                        preview.view === view
                          ? "bg-white text-slate-950"
                          : "text-slate-300 hover:text-white",
                      )}
                    >
                      {view} view
                    </button>
                  ))}
                </div>
                <button
                  type="button"
                  onClick={() => setPreview(null)}
                  className="flex h-9 w-9 items-center justify-center rounded-lg text-slate-300 transition hover:bg-white/10 hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-400"
                  aria-label="Close full X-ray preview"
                  autoFocus
                >
                  <X className="h-5 w-5" />
                </button>
              </div>
            </div>

            <div className="relative min-h-0 flex-1">
              <Image
                src={preview.view === "AP" ? preview.demo.apUrl! : preview.demo.latUrl}
                alt={`${preview.demo.label} ${preview.view} full X-ray`}
                fill
                priority
                sizes="100vw"
                className="object-contain p-2 sm:p-4"
              />
            </div>

            <div className="border-t border-white/10 px-4 py-2.5 text-center text-xs text-slate-400">
              {preview.demo.apUrl
                ? "Switch between AP and lateral views above · "
                : "Lateral view only · "}
              Press Esc or click outside to close
            </div>
          </div>
        </div>
      )}
    </>
  );
}
