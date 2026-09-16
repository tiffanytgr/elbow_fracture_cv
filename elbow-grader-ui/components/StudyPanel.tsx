"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  BarChart3,
  Check,
  Download,
  History,
  Lock,
  Pause,
  Play,
  Save,
  Timer as TimerIcon,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import { GradeConfidenceForm } from "@/components/GradeConfidenceForm";
import type { ConfidenceLevel, StudyMode } from "@/lib/studyTypes";

interface RecentEntry {
  reviewer: string | null;
  mode: string | null;
  case_id: string | null;
  pre_grade: string | null;
  post_grade: string | null;
  elapsed_hms: string;
  logged_at: string;
}

interface SummaryRow {
  group: string;
  count: number;
  mean_seconds: number;
  median_seconds: number;
  total_seconds: number;
}

interface StudyPanelProps {
  caseKey: string | null;
  caseId: string | null;
  inputMode: string;
  reviewer: string;
  mode: StudyMode;
  /** True once the AI result is visible on the page (AI arm only). */
  aiRevealed: boolean;
  /** The AI's own final grade / confidence, stored alongside the reader answers. */
  aiGrade?: string | null;
  aiConfidence?: number | null;
  /** Fires whenever the pre-AI answer lock state changes (gates Analyse + reveal). */
  onPreLockedChange: (locked: boolean) => void;
  /** Fires after a case is saved, with the caseKey that was saved. */
  onSaved: (caseKey: string) => void;
}

function formatHms(totalMs: number): string {
  const s = Math.max(0, Math.floor(totalMs / 1000));
  const hh = Math.floor(s / 3600);
  const mm = Math.floor((s % 3600) / 60);
  const ss = s % 60;
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${pad(hh)}:${pad(mm)}:${pad(ss)}`;
}

function formatSeconds(totalSeconds: number): string {
  const s = Math.max(0, Math.round(totalSeconds));
  const mm = Math.floor(s / 60);
  const ss = s % 60;
  return mm > 0 ? `${mm}m ${String(ss).padStart(2, "0")}s` : `${ss}s`;
}

export function StudyPanel({
  caseKey,
  caseId,
  inputMode,
  reviewer,
  mode,
  aiRevealed,
  aiGrade = null,
  aiConfidence = null,
  onPreLockedChange,
  onSaved,
}: StudyPanelProps) {
  const [running, setRunning] = useState(false);
  const [elapsedMs, setElapsedMs] = useState(0);

  const [preGrade, setPreGrade] = useState<string | null>(null);
  const [preConf, setPreConf] = useState<ConfidenceLevel | null>(null);
  const [preLocked, setPreLocked] = useState(false);
  const [postGrade, setPostGrade] = useState<string | null>(null);
  const [postConf, setPostConf] = useState<ConfidenceLevel | null>(null);

  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [savedAt, setSavedAt] = useState<string | null>(null);
  const [recent, setRecent] = useState<RecentEntry[]>([]);
  const [logPath, setLogPath] = useState<string | null>(null);
  const [summary, setSummary] = useState<SummaryRow[] | null>(null);
  const [showSummary, setShowSummary] = useState(false);

  const accumulatedRef = useRef(0);
  const segmentStartRef = useRef<number | null>(null);
  const startedAtRef = useRef<string | null>(null);

  // Keep the parent informed of the pre-AI lock so it can gate analysis/reveal.
  useEffect(() => {
    onPreLockedChange(preLocked);
  }, [preLocked, onPreLockedChange]);

  const loadRecent = useCallback(async () => {
    try {
      const res = await fetch("/api/study-log?limit=5", { cache: "no-store" });
      if (!res.ok) return;
      const data = await res.json();
      setRecent(Array.isArray(data.entries) ? data.entries : []);
      setLogPath(data.log_path ?? null);
    } catch {
      /* best-effort */
    }
  }, []);

  const loadSummary = useCallback(async () => {
    try {
      const res = await fetch("/api/study-log?summary=1&group_by=mode", {
        cache: "no-store",
      });
      if (!res.ok) return;
      const data = await res.json();
      setSummary(Array.isArray(data.summary) ? data.summary : []);
    } catch {
      /* best-effort */
    }
  }, []);

  useEffect(() => {
    loadRecent();
  }, [loadRecent]);

  useEffect(() => {
    if (showSummary) loadSummary();
  }, [showSummary, loadSummary]);

  // Auto reset & start whenever a new case is loaded; clear all answers.
  useEffect(() => {
    if (!caseKey) {
      setRunning(false);
      setElapsedMs(0);
      accumulatedRef.current = 0;
      segmentStartRef.current = null;
      startedAtRef.current = null;
    } else {
      accumulatedRef.current = 0;
      segmentStartRef.current = Date.now();
      startedAtRef.current = new Date().toISOString();
      setElapsedMs(0);
      setRunning(true);
    }
    setPreGrade(null);
    setPreConf(null);
    setPreLocked(false);
    setPostGrade(null);
    setPostConf(null);
    setSavedAt(null);
    setSaveError(null);
  }, [caseKey]);

  // Tick while running.
  useEffect(() => {
    if (!running) return;
    const id = window.setInterval(() => {
      const base = accumulatedRef.current;
      const seg =
        segmentStartRef.current !== null
          ? Date.now() - segmentStartRef.current
          : 0;
      setElapsedMs(base + seg);
    }, 250);
    return () => window.clearInterval(id);
  }, [running]);

  function pause() {
    if (!running) return;
    if (segmentStartRef.current !== null) {
      accumulatedRef.current += Date.now() - segmentStartRef.current;
      segmentStartRef.current = null;
    }
    setElapsedMs(accumulatedRef.current);
    setRunning(false);
  }

  function resume() {
    if (running || !caseKey) return;
    segmentStartRef.current = Date.now();
    if (!startedAtRef.current) startedAtRef.current = new Date().toISOString();
    setRunning(true);
  }

  async function saveCase() {
    if (!caseKey) return;
    setSaving(true);
    setSaveError(null);

    // Freeze the timer at the moment of saving.
    if (running && segmentStartRef.current !== null) {
      accumulatedRef.current += Date.now() - segmentStartRef.current;
      segmentStartRef.current = null;
    }
    const elapsedSeconds = accumulatedRef.current / 1000;
    setElapsedMs(accumulatedRef.current);
    setRunning(false);

    try {
      const res = await fetch("/api/study-log", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          reviewer: reviewer.trim() || null,
          mode,
          case_id: caseId,
          input_mode: inputMode,
          pre_grade: preGrade,
          pre_confidence: preConf,
          post_grade: mode === "ai" ? postGrade : null,
          post_confidence: mode === "ai" ? postConf : null,
          ai_grade: mode === "ai" ? aiGrade : null,
          ai_confidence: mode === "ai" ? aiConfidence : null,
          elapsed_seconds: elapsedSeconds,
          started_at: startedAtRef.current,
          ended_at: new Date().toISOString(),
        }),
      });
      const data = await res.json();
      if (!res.ok) {
        setSaveError(data.detail ?? data.error ?? `Error ${res.status}`);
        return;
      }
      setSavedAt(new Date().toLocaleTimeString());
      setLogPath(data.log_path ?? logPath);
      await loadRecent();
      if (showSummary) await loadSummary();
      onSaved(caseKey);
    } catch (e) {
      setSaveError(e instanceof Error ? e.message : "Failed to save");
    } finally {
      setSaving(false);
    }
  }

  const disabled = !caseKey;
  const alreadySaved = savedAt !== null;

  // Gate the save button per arm.
  const preComplete = preGrade !== null && preConf !== null;
  const postComplete = postGrade !== null && postConf !== null;
  const canSave =
    !disabled &&
    !saving &&
    !alreadySaved &&
    (mode === "control"
      ? preComplete
      : preLocked && aiRevealed && postComplete);

  let saveHint: string | null = null;
  if (!disabled && !alreadySaved) {
    if (mode === "control" && !preComplete) {
      saveHint = "Enter a grade and confidence to save.";
    } else if (mode === "ai" && !preLocked) {
      saveHint = "Lock your pre-AI read first.";
    } else if (mode === "ai" && !aiRevealed) {
      saveHint = "Run the AI analysis, then enter your post-AI read.";
    } else if (mode === "ai" && !postComplete) {
      saveHint = "Enter your post-AI grade and confidence to save.";
    }
  }

  return (
    <section className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <TimerIcon className="h-5 w-5 text-blue-600" />
          <h2 className="text-base font-semibold">
            Reader Assessment
            <span className="ml-2 rounded-full bg-slate-100 px-2 py-0.5 text-xs font-medium text-slate-600">
              {mode === "ai" ? "AI-assisted arm" : "Control arm"}
            </span>
          </h2>
        </div>
        <div className="flex items-center gap-3">
          <span className="font-mono text-3xl font-bold tabular-nums tracking-tight text-slate-900">
            {formatHms(elapsedMs)}
          </span>
          {running ? (
            <Button
              variant="outline"
              size="sm"
              onClick={pause}
              disabled={disabled}
              className="gap-1.5"
            >
              <Pause className="h-4 w-4" />
              Pause
            </Button>
          ) : (
            <Button
              variant="outline"
              size="sm"
              onClick={resume}
              disabled={disabled || alreadySaved}
              className="gap-1.5"
            >
              <Play className="h-4 w-4" />
              Resume
            </Button>
          )}
        </div>
      </div>

      <p className="mt-1 text-sm text-muted-foreground">
        {disabled
          ? "Load a case to start timing and grading."
          : alreadySaved
            ? "Saved. Load the next case to continue."
            : running
              ? "Stopwatch is running. Pause if you need to step away."
              : "Paused. Resume when you return to the case."}
      </p>

      {!disabled && (
        <div className="mt-4 space-y-4">
          {/* Pre-AI (or sole) reader answer */}
          <div className="rounded-lg border border-slate-200 bg-slate-50/60 p-4">
            <div className="mb-3 flex items-center gap-2">
              <span className="text-sm font-semibold text-slate-800">
                {mode === "ai" ? "Your read — before AI" : "Your read"}
              </span>
              {mode === "ai" && preLocked && (
                <span className="inline-flex items-center gap-1 rounded-full bg-green-100 px-2 py-0.5 text-xs font-medium text-green-800">
                  <Lock className="h-3 w-3" />
                  Locked
                </span>
              )}
            </div>
            <GradeConfidenceForm
              idPrefix="pre"
              grade={preGrade}
              confidence={preConf}
              onGradeChange={setPreGrade}
              onConfidenceChange={setPreConf}
              disabled={mode === "ai" && preLocked}
            />
            {mode === "ai" && !preLocked && (
              <div className="mt-3">
                <Button
                  size="sm"
                  variant="outline"
                  disabled={!preComplete}
                  onClick={() => setPreLocked(true)}
                  className="gap-1.5"
                >
                  <Lock className="h-4 w-4" />
                  Lock pre-AI read
                </Button>
                <span className="ml-2 text-xs text-slate-500">
                  Locking reveals the AI analysis; the pre-AI answer can’t be changed afterwards.
                </span>
              </div>
            )}
          </div>

          {/* Post-AI reader answer (AI arm only, after the result is shown) */}
          {mode === "ai" && preLocked && (
            <div
              className={`rounded-lg border p-4 ${
                aiRevealed
                  ? "border-slate-200 bg-white"
                  : "border-dashed border-slate-200 bg-slate-50/40"
              }`}
            >
              <div className="mb-3 text-sm font-semibold text-slate-800">
                Your read — after AI
              </div>
              {aiRevealed ? (
                <GradeConfidenceForm
                  idPrefix="post"
                  grade={postGrade}
                  confidence={postConf}
                  onGradeChange={setPostGrade}
                  onConfidenceChange={setPostConf}
                />
              ) : (
                <p className="text-sm text-muted-foreground">
                  Run the AI analysis below, review it, then record your final read here.
                </p>
              )}
            </div>
          )}
        </div>
      )}

      <div className="mt-4 flex flex-wrap items-center gap-3">
        <Button
          size="sm"
          onClick={saveCase}
          disabled={!canSave}
          className="gap-1.5 bg-gradient-to-r from-[#1e3a5f] to-[#2563a8] hover:from-[#1e3a5f]/90 hover:to-[#2563a8]/90"
        >
          <Save className="h-4 w-4" />
          {saving ? "Saving…" : "Save assessment"}
        </Button>
        {savedAt && !saveError && (
          <span className="inline-flex items-center gap-1 text-sm font-medium text-green-700">
            <Check className="h-4 w-4" />
            Saved at {savedAt}
          </span>
        )}
        {saveHint && (
          <span className="text-sm text-amber-700">{saveHint}</span>
        )}
      </div>

      {saveError && (
        <p className="mt-3 rounded-md bg-red-50 px-3 py-2 text-sm text-red-800">
          <strong>Could not save:</strong> {saveError}
        </p>
      )}

      {recent.length > 0 && (
        <div className="mt-5 border-t border-slate-100 pt-4">
          <div className="mb-4 flex flex-wrap items-center gap-2">
            <a
              href="/api/study-log?format=csv"
              className="inline-flex items-center gap-1.5 rounded-md border border-slate-300 px-3 py-1.5 text-sm font-medium text-slate-700 transition hover:border-blue-400 hover:text-blue-700"
            >
              <Download className="h-4 w-4" />
              Download log (CSV)
            </a>
            <a
              href="/api/study-log?summary=1&format=csv&group_by=mode"
              className="inline-flex items-center gap-1.5 rounded-md border border-slate-300 px-3 py-1.5 text-sm font-medium text-slate-700 transition hover:border-blue-400 hover:text-blue-700"
            >
              <Download className="h-4 w-4" />
              Summary (CSV)
            </a>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setShowSummary((v) => !v)}
              className="gap-1.5"
            >
              <BarChart3 className="h-4 w-4" />
              {showSummary ? "Hide summary" : "View summary"}
            </Button>
          </div>

          {showSummary && summary && summary.length > 0 && (
            <div className="mb-4 overflow-x-auto rounded-lg border border-slate-200">
              <table className="w-full text-sm">
                <thead className="bg-slate-50 text-left text-xs uppercase tracking-wide text-slate-500">
                  <tr>
                    <th className="px-3 py-2 font-semibold">Arm</th>
                    <th className="px-3 py-2 text-right font-semibold">Cases</th>
                    <th className="px-3 py-2 text-right font-semibold">Mean</th>
                    <th className="px-3 py-2 text-right font-semibold">Median</th>
                    <th className="px-3 py-2 text-right font-semibold">Total</th>
                  </tr>
                </thead>
                <tbody>
                  {summary.map((row) => {
                    const isAll = row.group === "ALL";
                    const label =
                      row.group === "ai"
                        ? "AI-assisted"
                        : row.group === "control"
                          ? "Control"
                          : isAll
                            ? "All cases"
                            : row.group;
                    return (
                      <tr
                        key={row.group}
                        className={
                          isAll
                            ? "border-t border-slate-200 bg-slate-50 font-semibold"
                            : "border-t border-slate-100"
                        }
                      >
                        <td className="px-3 py-2">{label}</td>
                        <td className="px-3 py-2 text-right tabular-nums">
                          {row.count}
                        </td>
                        <td className="px-3 py-2 text-right tabular-nums">
                          {formatSeconds(row.mean_seconds)}
                        </td>
                        <td className="px-3 py-2 text-right tabular-nums">
                          {formatSeconds(row.median_seconds)}
                        </td>
                        <td className="px-3 py-2 text-right tabular-nums">
                          {formatSeconds(row.total_seconds)}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}

          <div className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-slate-500">
            <History className="h-3.5 w-3.5" />
            Recently logged
          </div>
          <ul className="mt-2 space-y-1.5">
            {recent.map((entry, i) => (
              <li
                key={`${entry.logged_at}-${i}`}
                className="flex flex-wrap items-center justify-between gap-2 text-sm"
              >
                <span className="text-slate-700">
                  <span className="font-mono font-semibold">
                    {entry.elapsed_hms}
                  </span>{" "}
                  · {entry.case_id ?? "—"} ·{" "}
                  {entry.mode === "ai" ? "AI" : "Control"}
                  {entry.pre_grade ? ` · pre ${entry.pre_grade}` : ""}
                  {entry.post_grade ? ` → post ${entry.post_grade}` : ""}
                  {entry.reviewer ? ` · ${entry.reviewer}` : ""}
                </span>
                <span className="text-xs text-slate-400">
                  {new Date(entry.logged_at).toLocaleString()}
                </span>
              </li>
            ))}
          </ul>
          {logPath && (
            <p className="mt-3 break-all text-xs text-slate-400">
              Saved on this device: <code>{logPath}</code>
            </p>
          )}
        </div>
      )}
    </section>
  );
}
