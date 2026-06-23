"use client";

import { useState } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";
import { Separator } from "@/components/ui/separator";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import type { GraderConfig } from "@/lib/types";

interface SidebarProps {
  modelStatus: Record<string, string> | null;
  config: GraderConfig;
  onConfigChange: (cfg: GraderConfig) => void;
  device?: string;
}

function StatusDot({ status }: { status: string }) {
  const lower = status.toLowerCase();
  const ok = lower.includes("ok") || lower.includes("loaded");
  return (
    <span
      className={`inline-block w-2.5 h-2.5 rounded-full mr-2 flex-shrink-0 ${ok ? "bg-green-500" : "bg-red-500"}`}
    />
  );
}

export function Sidebar({ modelStatus, config, onConfigChange, device }: SidebarProps) {
  const [advancedOpen, setAdvancedOpen] = useState(false);

  return (
    <aside className="w-64 flex-shrink-0 bg-slate-50 border-r border-border p-4 flex flex-col gap-4 min-h-screen">
      {/* System status */}
      <div>
        <p className="text-sm font-semibold text-slate-700 mb-2">System status</p>
        {modelStatus ? (
          <ul className="space-y-1.5">
            {Object.entries(modelStatus).map(([name, status]) => (
              <li key={name} className="flex items-center text-xs text-slate-600">
                <StatusDot status={status} />
                <span className="font-medium mr-1">{name}</span>
                <span className="text-slate-400">— {status}</span>
              </li>
            ))}
          </ul>
        ) : (
          <p className="text-xs text-slate-400 italic">Run an analysis to check model readiness</p>
        )}
      </div>

      <Separator />

      {/* Minimum confidence */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <p className="text-sm font-semibold text-slate-700">Minimum confidence</p>
          <span className="text-sm font-mono text-primary">
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
        <p className="text-xs text-muted-foreground">
          Results below this confidence level will be withheld. Lower = more results shown; higher = stricter.
        </p>
      </div>

      <Separator />

      {/* Advanced settings (collapsed by default) */}
      <div>
        <button
          type="button"
          onClick={() => setAdvancedOpen((v) => !v)}
          className="flex items-center justify-between w-full text-sm font-semibold text-slate-700"
        >
          <span>Advanced settings</span>
          {advancedOpen
            ? <ChevronDown className="w-4 h-4 text-slate-400" />
            : <ChevronRight className="w-4 h-4 text-slate-400" />}
        </button>

        {advancedOpen && (
          <div className="mt-3 space-y-4">
            <label className="flex items-center justify-between gap-2 cursor-pointer">
              <div>
                <p className="text-sm font-medium">Auto-orient lateral X-ray</p>
                <p className="text-xs text-muted-foreground">
                  Enable for raw clinical DICOM exports that may be rotated or flipped
                </p>
              </div>
              <Switch
                checked={config.runFullLatAlignment}
                onCheckedChange={(v) =>
                  onConfigChange({ ...config, runFullLatAlignment: v })
                }
              />
            </label>

            <label className="flex items-center justify-between gap-2 cursor-pointer">
              <div>
                <p className="text-sm font-medium">Bone analysis (AHL)</p>
                <p className="text-xs text-muted-foreground">
                  Segments bone on the lateral view and measures the Anterior Humeral Line
                </p>
              </div>
              <Switch
                checked={config.runSam2}
                onCheckedChange={(v) => onConfigChange({ ...config, runSam2: v })}
              />
            </label>
          </div>
        )}
      </div>

      {device && (
        <>
          <Separator />
          <p className="text-xs text-muted-foreground">
            Device: <code className="font-mono">{device}</code>
          </p>
        </>
      )}
    </aside>
  );
}
