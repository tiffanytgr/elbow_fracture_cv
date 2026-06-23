"use client";

import { useState } from "react";
import { Download } from "lucide-react";
import { Button } from "@/components/ui/button";
import type { PredictResponse } from "@/lib/types";

interface AuditTabProps {
  result: PredictResponse;
}

export function AuditTab({ result }: AuditTabProps) {
  const [showFull, setShowFull] = useState(false);

  function downloadJson() {
    const blob = new Blob([JSON.stringify(result.result_json, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "prediction.json";
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div className="py-4 space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Image hashes */}
        <div>
          <p className="text-sm font-semibold mb-2">Image hashes</p>
          <pre className="text-xs bg-slate-50 rounded-lg border border-border p-3 font-mono">
            {`AP SHA1 : ${result.ap_sha1 ?? "n/a"}\nLAT SHA1: ${result.lat_sha1 ?? "n/a"}`}
          </pre>
        </div>

        {/* Config snapshot */}
        <div>
          <p className="text-sm font-semibold mb-2">Config snapshot</p>
          <pre className="text-xs bg-slate-50 rounded-lg border border-border p-3 overflow-x-auto max-h-40">
            {JSON.stringify(result.config_snapshot, null, 2)}
          </pre>
        </div>
      </div>

      {/* Full audit toggle */}
      <div>
        <button
          type="button"
          onClick={() => setShowFull((v) => !v)}
          className="text-sm text-primary font-medium hover:underline"
        >
          {showFull ? "Hide" : "Show"} full result JSON
        </button>
        {showFull && (
          <pre className="mt-2 text-xs bg-slate-50 rounded-lg border border-border p-3 overflow-x-auto max-h-96">
            {JSON.stringify(result.result_json, null, 2)}
          </pre>
        )}
      </div>

      <Button onClick={downloadJson} variant="outline" size="sm" className="gap-2">
        <Download className="w-4 h-4" />
        Download result JSON
      </Button>
    </div>
  );
}
