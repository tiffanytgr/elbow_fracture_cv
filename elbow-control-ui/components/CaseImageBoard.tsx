"use client";

import { useEffect, useMemo, useState } from "react";
import Image from "next/image";
import { Maximize2, X } from "lucide-react";

interface CaseImageBoardProps {
  apFile: File | null;
  latFile: File | null;
}

interface ViewItem {
  label: "AP" | "LAT";
  url: string;
}

/** Read-only AP/LAT viewer with click-to-expand, for manual grading. */
export function CaseImageBoard({ apFile, latFile }: CaseImageBoardProps) {
  const [apUrl, setApUrl] = useState<string | null>(null);
  const [latUrl, setLatUrl] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<ViewItem | null>(null);

  useEffect(() => {
    if (!apFile) {
      setApUrl(null);
      return;
    }
    const url = URL.createObjectURL(apFile);
    setApUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [apFile]);

  useEffect(() => {
    if (!latFile) {
      setLatUrl(null);
      return;
    }
    const url = URL.createObjectURL(latFile);
    setLatUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [latFile]);

  const views = useMemo<ViewItem[]>(() => {
    const list: ViewItem[] = [];
    if (apUrl) list.push({ label: "AP", url: apUrl });
    if (latUrl) list.push({ label: "LAT", url: latUrl });
    return list;
  }, [apUrl, latUrl]);

  useEffect(() => {
    if (!expanded) return;
    const previousOverflow = document.body.style.overflow;
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setExpanded(null);
    };
    document.body.style.overflow = "hidden";
    window.addEventListener("keydown", closeOnEscape);
    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener("keydown", closeOnEscape);
    };
  }, [expanded]);

  if (views.length === 0) {
    return (
      <p className="text-sm text-muted-foreground">No images loaded.</p>
    );
  }

  return (
    <>
      <div
        className={`grid gap-3 ${views.length > 1 ? "sm:grid-cols-2" : "grid-cols-1"}`}
      >
        {views.map((view) => (
          <button
            key={view.label}
            type="button"
            onClick={() => setExpanded(view)}
            className="group relative block overflow-hidden rounded-lg border border-slate-200 bg-slate-950 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-2"
            aria-label={`Expand ${view.label} X-ray`}
          >
            <Image
              src={view.url}
              alt={`${view.label} X-ray`}
              width={640}
              height={460}
              unoptimized
              className="h-[320px] w-full object-contain transition-transform group-hover:scale-[1.02] lg:h-[420px]"
            />
            <span className="absolute bottom-2 left-2 rounded bg-black/70 px-2 py-0.5 text-xs font-semibold text-white">
              {view.label}
            </span>
            <span className="absolute bottom-2 right-2 flex h-8 w-8 items-center justify-center rounded-md bg-black/65 text-white opacity-0 transition-opacity group-hover:opacity-100 group-focus-visible:opacity-100">
              <Maximize2 className="h-4 w-4" />
            </span>
          </button>
        ))}
      </div>

      {expanded && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/90 p-3 backdrop-blur-sm sm:p-6"
          onMouseDown={(event) => {
            if (event.target === event.currentTarget) setExpanded(null);
          }}
        >
          <div
            role="dialog"
            aria-modal="true"
            aria-label={`${expanded.label} full X-ray`}
            className="flex h-[min(92vh,900px)] w-full max-w-7xl flex-col overflow-hidden rounded-xl border border-white/15 bg-slate-950 shadow-2xl"
          >
            <div className="flex items-center justify-between gap-3 border-b border-white/10 px-4 py-3 sm:px-5">
              <h3 className="font-semibold text-white">
                {expanded.label} view
              </h3>
              <div className="flex items-center gap-2">
                {views.length > 1 && (
                  <div className="flex rounded-lg bg-white/10 p-1">
                    {views.map((view) => (
                      <button
                        key={view.label}
                        type="button"
                        onClick={() => setExpanded(view)}
                        aria-pressed={expanded.label === view.label}
                        className={`rounded-md px-3 py-1.5 text-xs font-semibold transition ${
                          expanded.label === view.label
                            ? "bg-white text-slate-950"
                            : "text-slate-300 hover:text-white"
                        }`}
                      >
                        {view.label} view
                      </button>
                    ))}
                  </div>
                )}
                <button
                  type="button"
                  onClick={() => setExpanded(null)}
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
                src={expanded.url}
                alt={`${expanded.label} full X-ray`}
                fill
                unoptimized
                sizes="100vw"
                className="object-contain p-2 sm:p-4"
              />
            </div>
            <div className="border-t border-white/10 px-4 py-2.5 text-center text-xs text-slate-400">
              Press Esc or click outside to close
            </div>
          </div>
        </div>
      )}
    </>
  );
}
