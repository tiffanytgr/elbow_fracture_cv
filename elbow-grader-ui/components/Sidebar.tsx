"use client";

import { useState } from "react";
import { ChevronDown, ChevronRight, Download, FileText, Settings, ShieldCheck } from "lucide-react";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import type { GraderConfig, PredictResponse } from "@/lib/types";

interface SidebarProps {
  modelStatus: Record<string, string> | null;
  config: GraderConfig;
  onConfigChange: (cfg: GraderConfig) => void;
  result: PredictResponse | null;
  device?: string;
}

function StatusDot({ status }: { status: string }) {
  const lower = status.toLowerCase();
  const ok = lower.includes("ok") || lower.includes("loaded");
  return (
    <span
      className={`mr-2 inline-block h-2.5 w-2.5 flex-shrink-0 rounded-full ${
        ok ? "bg-emerald-500" : "bg-red-500"
      }`}
    />
  );
}

export function Sidebar({ modelStatus, config, onConfigChange, result, device }: SidebarProps) {
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const ahlDetails = result?.geometric?.ahl_diagnostic ?? null;

  function downloadAuditJson() {
    if (!result) return;
    const blob = new Blob([JSON.stringify(result.result_json, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "prediction.json";
    anchor.click();
    URL.revokeObjectURL(url);
  }

  return (
    <aside className="flex min-h-screen w-full flex-col border-b border-slate-200 bg-gradient-to-b from-slate-50 to-white p-4 shadow-[4px_0_18px_rgba(15,23,42,0.04)] lg:sticky lg:top-0 lg:h-screen lg:w-72 lg:flex-shrink-0 lg:overflow-y-auto lg:border-b-0 lg:border-r">
      <div className="mb-5 px-2 py-2">
        <p className="text-xl font-bold tracking-tight text-[#102a56]">Elbow AI</p>
        <p className="text-sm font-medium text-slate-500">KKH AI Research</p>
      </div>

      <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
        <p className="mb-3 text-xs font-bold uppercase tracking-wider text-slate-600">
          System status
        </p>
        {modelStatus ? (
          <ul className="space-y-2.5">
            {Object.entries(modelStatus).map(([name, status]) => (
              <li key={name} className="flex items-center text-xs text-slate-600">
                <StatusDot status={status} />
                <span className="min-w-0 flex-1 truncate font-medium">{name}</span>
                <span className="ml-2 text-emerald-600">{status}</span>
              </li>
            ))}
          </ul>
        ) : (
          <p className="text-xs leading-5 text-slate-400">
            Run an analysis to check model readiness.
          </p>
        )}
      </div>

      <div className="mt-4 space-y-3 rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
        <div className="flex items-center justify-between">
          <p className="text-xs font-bold uppercase tracking-wider text-slate-600">
            Minimum confidence
          </p>
          <span className="text-sm font-bold text-blue-700">
            {Math.round(config.confidenceThreshold * 100)}%
          </span>
        </div>
        <Slider
          min={0.5}
          max={0.95}
          step={0.05}
          value={[config.confidenceThreshold]}
          onValueChange={([v]) =>
            onConfigChange({ ...config, confidenceThreshold: v })
          }
        />
        <p className="text-xs leading-5 text-slate-500">
          Results below this confidence level will be withheld. Lower = more results shown; higher = stricter.
        </p>
      </div>

      <div className="mt-4 rounded-xl border border-slate-200 bg-white shadow-sm">
        <button
          type="button"
          onClick={() => setAdvancedOpen((v) => !v)}
          className="flex w-full items-center justify-between rounded-xl px-4 py-4 text-xs font-bold uppercase tracking-wider text-slate-600 hover:bg-slate-50"
        >
          <span className="flex items-center gap-2">
            <Settings className="h-4 w-4 text-blue-600" />
            Advanced settings
          </span>
          {advancedOpen ? (
            <ChevronDown className="h-4 w-4 text-slate-400" />
          ) : (
            <ChevronRight className="h-4 w-4 text-slate-400" />
          )}
        </button>

        {advancedOpen && (
          <div className="space-y-4 border-t border-slate-100 px-4 py-4">
            <label className="flex cursor-pointer items-center justify-between gap-2">
              <div>
                <p className="text-sm font-medium">Auto-orient lateral X-ray</p>
                <p className="text-xs text-muted-foreground">
                  Enable for raw clinical exports that may be rotated or flipped
                </p>
              </div>
              <Switch
                checked={config.runFullLatAlignment}
                onCheckedChange={(v) =>
                  onConfigChange({ ...config, runFullLatAlignment: v })
                }
              />
            </label>

            <label className="flex cursor-pointer items-center justify-between gap-2">
              <div>
                <p className="text-sm font-medium">Bone analysis (AHL)</p>
                <p className="text-xs text-muted-foreground">
                  Segment bone and measure the Anterior Humeral Line
                </p>
              </div>
              <Switch
                checked={config.runSam2}
                onCheckedChange={(v) => onConfigChange({ ...config, runSam2: v })}
              />
            </label>

            {result && (
              <div className="space-y-2 border-t border-slate-200 pt-4">
                <p className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-slate-500">
                  <FileText className="h-3.5 w-3.5" />
                  Technical details
                </p>

                {result.log.length > 0 && (
                  <details className="rounded-lg border border-slate-200 bg-slate-50">
                    <summary className="cursor-pointer select-none px-3 py-2 text-xs font-medium text-slate-700">
                      Technical log ({result.log.length} lines)
                    </summary>
                    <div className="max-h-48 overflow-auto border-t border-slate-200 p-3">
                      {result.log.map((line, index) => (
                        <p key={index} className="break-words font-mono text-[10px] leading-4 text-slate-600">
                          {line}
                        </p>
                      ))}
                    </div>
                  </details>
                )}

                {ahlDetails && Object.keys(ahlDetails).length > 0 && (
                  <details className="rounded-lg border border-slate-200 bg-slate-50">
                    <summary className="cursor-pointer select-none px-3 py-2 text-xs font-medium text-slate-700">
                      AHL details
                    </summary>
                    <pre className="max-h-48 overflow-auto whitespace-pre-wrap break-all border-t border-slate-200 p-3 text-[10px] leading-4 text-slate-600">
                      {JSON.stringify(
                        Object.fromEntries(
                          Object.entries(ahlDetails).filter(([key]) =>
                            ["zone", "method", "ahl_x_at_cap", "dist_to_ahl_px", "cap_radius_px"].includes(key),
                          ),
                        ),
                        null,
                        2,
                      )}
                    </pre>
                  </details>
                )}

                <details className="rounded-lg border border-slate-200 bg-slate-50">
                  <summary className="cursor-pointer select-none px-3 py-2 text-xs font-medium text-slate-700">
                    Audit trail
                  </summary>
                  <div className="space-y-3 border-t border-slate-200 p-3">
                    <div>
                      <p className="mb-1 text-[10px] font-semibold uppercase text-slate-500">Image hashes</p>
                      <pre className="overflow-x-auto whitespace-pre-wrap break-all text-[10px] leading-4 text-slate-600">
                        {`AP: ${result.ap_sha1 ?? "n/a"}\nLAT: ${result.lat_sha1 ?? "n/a"}`}
                      </pre>
                    </div>
                    <div>
                      <p className="mb-1 text-[10px] font-semibold uppercase text-slate-500">Configuration</p>
                      <pre className="max-h-40 overflow-auto whitespace-pre-wrap break-all text-[10px] leading-4 text-slate-600">
                        {JSON.stringify(result.config_snapshot, null, 2)}
                      </pre>
                    </div>
                    <button
                      type="button"
                      onClick={downloadAuditJson}
                      className="flex w-full items-center justify-center gap-1.5 rounded-md border border-slate-300 bg-white px-2 py-2 text-xs font-medium text-slate-700 hover:bg-slate-100"
                    >
                      <Download className="h-3.5 w-3.5" />
                      Download result JSON
                    </button>
                  </div>
                </details>
              </div>
            )}
          </div>
        )}
      </div>

      {device && (
        <p className="mt-4 px-1 text-xs text-muted-foreground">
          Device: <code className="font-mono">{device}</code>
        </p>
      )}

      <div className="mt-auto hidden items-start gap-2 px-3 pb-2 pt-8 text-xs leading-5 text-slate-500 lg:flex">
        <ShieldCheck className="mt-0.5 h-4 w-4 shrink-0 text-blue-600" />
        <span>
          <strong className="font-semibold text-slate-600">
            Research prototype — not for clinical use.
          </strong>{" "}
          Results must be interpreted by a qualified clinician and do not
          replace radiological review.
        </span>
      </div>
    </aside>
  );
}
