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
  Send,
  Timer as TimerIcon,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  ConfidencePicker,
  GradeConfidenceForm,
  GradePicker,
} from "@/components/GradeConfidenceForm";
import type { ConfidenceLevel } from "@/lib/studyTypes";

interface RecentEntry {
  reviewer: string | null;
  case_id: string | null;
  pre_grade: string | null;
  post_grade: string | null;
  decision_time_seconds: number | null;
  elapsed_hms: string;
  logged_at: string;
}

interface SummaryRow {
  group: string;
  count: number;
  mean_seconds: number;
  median_seconds: number;
  total_seconds: number;
  mean_decision_seconds: number | null;
  median_decision_seconds: number | null;
}

interface StudyPanelProps {
  caseKey: string | null;
  /** Case identifier entered by the reader; answers are blocked until set. */
  caseId: string | null;
  apPath: string | null;
  latPath: string | null;
  inputMode: string;
  reviewer: string;
  /** True once the AI result is visible on the page. */
  aiRevealed: boolean;
  /** The AI's outputs, stored alongside the reader answers. */
  aiGartlandGrade?: string | null;
  aiCnnGrade?: string | null;
  aiGeometricGrade?: string | null;
  aiConfidence?: number | null;
  /** Backend model run time for the AI result shown, in seconds. */
  aiProcessingSeconds?: number | null;
  /** Where the backend writes predictions.log (AI arm only). */
  predictionLogPath?: string | null;
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
  apPath,
  latPath,
  inputMode,
  reviewer,
  aiRevealed,
  aiGartlandGrade = null,
  aiCnnGrade = null,
  aiGeometricGrade = null,
  aiConfidence = null,
  aiProcessingSeconds = null,
  predictionLogPath = null,
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
  const [comments, setComments] = useState("");

  // Two-step submission: (1) the decision grade, which stops the decision
  // clock; (2) confidence + follow-up questions, which saves the case.
  const [gradeSubmitted, setGradeSubmitted] = useState(false);
  const [decisionSeconds, setDecisionSeconds] = useState<number | null>(null);

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
  // Stopwatch reading (ms) and wall-clock time when the AI results appeared,
  // i.e. when the decision window opened.
  const decisionStartMsRef = useRef<number | null>(null);
  const decisionStartedAtRef = useRef<string | null>(null);
  const gradeSubmittedAtRef = useRef<string | null>(null);

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
      const res = await fetch("/api/study-log?summary=1&group_by=reviewer", {
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

  function currentElapsedMs(): number {
    const seg =
      segmentStartRef.current !== null ? Date.now() - segmentStartRef.current : 0;
    return accumulatedRef.current + seg;
  }

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
    decisionStartMsRef.current = null;
    decisionStartedAtRef.current = null;
    gradeSubmittedAtRef.current = null;
    setPreGrade(null);
    setPreConf(null);
    setPreLocked(false);
    setPostGrade(null);
    setPostConf(null);
    setComments("");
    setGradeSubmitted(false);
    setDecisionSeconds(null);
    setSavedAt(null);
    setSaveError(null);
  }, [caseKey]);

  // The decision window opens when the AI result is shown. A re-run
  // before the grade is submitted restarts it from the latest reveal.
  useEffect(() => {
    if (gradeSubmitted) return;
    if (aiRevealed) {
      decisionStartMsRef.current = currentElapsedMs();
      decisionStartedAtRef.current = new Date().toISOString();
    } else {
      decisionStartMsRef.current = null;
      decisionStartedAtRef.current = null;
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [aiRevealed]);

  // Tick while running.
  useEffect(() => {
    if (!running) return;
    const id = window.setInterval(() => {
      setElapsedMs(currentElapsedMs());
    }, 250);
    return () => window.clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
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

  /** Submit 1: lock the decision grade and record the decision time. */
  function submitGrade() {
    if (!canSubmitGrade || decisionStartMsRef.current === null) return;
    const seconds = (currentElapsedMs() - decisionStartMsRef.current) / 1000;
    gradeSubmittedAtRef.current = new Date().toISOString();
    setDecisionSeconds(Math.max(0, seconds));
    setGradeSubmitted(true);
  }

  /** Submit 2: confidence + follow-up answers; writes the case to the log. */
  async function saveCase() {
    if (!caseKey || !canSave) return;
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
          case_id: caseId,
          ap_path: apPath,
          lat_path: latPath,
          input_mode: inputMode,
          pre_grade: preGrade,
          pre_confidence: preConf,
          post_grade: postGrade,
          post_confidence: postConf,
          ai_gartland_grade: aiGartlandGrade,
          ai_cnn_grade: aiCnnGrade,
          ai_geometric_grade: aiGeometricGrade,
          ai_confidence: aiConfidence,
          ai_processing_time_seconds: aiProcessingSeconds,
          notes: comments,
          decision_time_seconds: decisionSeconds,
          decision_started_at: decisionStartedAtRef.current,
          grade_submitted_at: gradeSubmittedAtRef.current,
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
  // Answers stay locked until the reader has entered a case ID.
  const needsCaseId = !caseId && !alreadySaved;

  const preComplete = preGrade !== null && preConf !== null;
  const canSubmitGrade =
    !disabled &&
    !needsCaseId &&
    !gradeSubmitted &&
    postGrade !== null &&
    preLocked &&
    aiRevealed;
  const canSave =
    !disabled &&
    !needsCaseId &&
    !saving &&
    !alreadySaved &&
    gradeSubmitted &&
    postConf !== null &&
    aiRevealed;

  let hint: string | null = null;
  if (!disabled && !alreadySaved) {
    if (needsCaseId) {
      hint = "Enter a case ID above to start grading.";
    } else if (!preLocked) {
      hint = "Lock your pre-AI read first.";
    } else if (!aiRevealed) {
      hint = gradeSubmitted
        ? "AI result is out of date — re-run the analysis to submit."
        : "Run the AI analysis, then choose your grade.";
    } else if (!gradeSubmitted) {
      hint = postGrade
        ? "Submit your grade to record your decision time."
        : "Choose a Gartland grade.";
    } else if (!postConf) {
      hint = "Rate your confidence, then submit the assessment.";
    }
  }

  return (
    <section className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <TimerIcon className="h-5 w-5 text-blue-600" />
          <h2 className="text-base font-semibold">
            Reader Assessment
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
            ? "Saved. Upload or select new X-rays to start the next case."
            : running
              ? "Stopwatch is running. Pause if you need to step away."
              : "Paused. Resume when you return to the case."}
      </p>
      {caseId && (
        <p className="mt-1 break-all text-xs text-slate-500">
          Case: <code>{caseId}</code>
        </p>
      )}

      {!disabled && (
        <div className="mt-4 space-y-4">
          {/* Pre-AI read */}
        <div className="rounded-lg border border-slate-200 bg-slate-50/60 p-4">
            <div className="mb-3 flex items-center gap-2">
              <span className="text-sm font-semibold text-slate-800">
                Your read — before AI
              </span>
              {preLocked && (
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
              disabled={preLocked || needsCaseId}
            />
            {!preLocked && (
              <div className="mt-3">
                <Button
                  size="sm"
                  variant="outline"
                  disabled={!preComplete || needsCaseId}
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

          {/* Step 1 — post-AI decision grade */}
          {preLocked && (
            <div
              className={`rounded-lg border p-4 ${
                aiRevealed || gradeSubmitted
                  ? "border-slate-200 bg-white"
                  : "border-dashed border-slate-200 bg-slate-50/40"
              }`}
            >
              <div className="mb-3 flex flex-wrap items-center gap-2">
                <span className="text-sm font-semibold text-slate-800">
                  Step 1 — Your grade after AI
                </span>
                {gradeSubmitted && (
                  <span className="inline-flex items-center gap-1 rounded-full bg-green-100 px-2 py-0.5 text-xs font-medium text-green-800">
                    <Check className="h-3 w-3" />
                    Submitted
                    {decisionSeconds !== null &&
                      ` · decision time ${formatSeconds(decisionSeconds)}`}
                  </span>
                )}
              </div>
              {!aiRevealed && !gradeSubmitted ? (
                <p className="text-sm text-muted-foreground">
                  Run the AI analysis below and review it, then choose your grade here.
                </p>
              ) : (
                <>
                  <GradePicker
                    idPrefix="decision"
                    grade={postGrade}
                    onGradeChange={setPostGrade}
                    disabled={gradeSubmitted || needsCaseId}
                  />
                  {!gradeSubmitted && (
                    <div className="mt-3 flex flex-wrap items-center gap-2">
                      <Button
                        size="sm"
                        onClick={submitGrade}
                        disabled={!canSubmitGrade}
                        className="gap-1.5 bg-gradient-to-r from-[#1e3a5f] to-[#2563a8] hover:from-[#1e3a5f]/90 hover:to-[#2563a8]/90"
                      >
                        <Send className="h-4 w-4" />
                        Submit grade
                      </Button>
                      <span className="text-xs text-slate-500">
                        Decision time runs from when the AI results appear until you submit your grade.
                      </span>
                    </div>
                  )}
                </>
              )}
            </div>
          )}

          {/* Step 2 — confidence + follow-up questions, then save */}
          {gradeSubmitted && (
            <div className="rounded-lg border border-slate-200 bg-white p-4">
              <div className="mb-3 text-sm font-semibold text-slate-800">
                Step 2 — Confidence and follow-up
              </div>
              <div className="space-y-4">
                <ConfidencePicker
                  idPrefix="decision"
                  confidence={postConf}
                  onConfidenceChange={setPostConf}
                  disabled={alreadySaved}
                />
                <label className="block text-sm">
                  <span className="mb-1.5 block font-medium text-slate-700">
                    Comments (optional)
                  </span>
                  <textarea
                    value={comments}
                    onChange={(e) => setComments(e.target.value)}
                    disabled={alreadySaved}
                    rows={2}
                    placeholder="Anything notable about this case or the AI output"
                    className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500 disabled:cursor-not-allowed disabled:opacity-60"
                  />
                </label>
              </div>
              <div className="mt-4">
                <Button
                  size="sm"
                  onClick={saveCase}
                  disabled={!canSave}
                  className="gap-1.5 bg-gradient-to-r from-[#1e3a5f] to-[#2563a8] hover:from-[#1e3a5f]/90 hover:to-[#2563a8]/90"
                >
                  <Send className="h-4 w-4" />
                  {saving ? "Submitting…" : "Submit assessment"}
                </Button>
              </div>
            </div>
          )}
        </div>
      )}

      {(savedAt || hint) && (
        <div className="mt-4 flex flex-wrap items-center gap-3">
          {savedAt && !saveError && (
            <span className="inline-flex items-center gap-1 text-sm font-medium text-green-700">
              <Check className="h-4 w-4" />
              Saved at {savedAt}
            </span>
          )}
          {hint && <span className="text-sm text-amber-700">{hint}</span>}
        </div>
      )}

      {savedAt && !saveError && logPath && (
        <div className="mt-2 rounded-md bg-green-50 px-3 py-2 text-xs text-green-900">
          <p className="break-all">
            Assessment log: <code>{logPath}</code>
          </p>
          {predictionLogPath && (
            <p className="mt-1 break-all">
              AI prediction log (backend machine): <code>{predictionLogPath}</code>
            </p>
          )}
        </div>
      )}

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
              href="/api/study-log?summary=1&format=csv&group_by=reviewer"
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
                    <th className="px-3 py-2 font-semibold">Reviewer</th>
                    <th className="px-3 py-2 text-right font-semibold">Cases</th>
                    <th className="px-3 py-2 text-right font-semibold">Mean</th>
                    <th className="px-3 py-2 text-right font-semibold">Median</th>
                    <th className="px-3 py-2 text-right font-semibold">Total</th>
                    <th className="px-3 py-2 text-right font-semibold">Mean decision</th>
                    <th className="px-3 py-2 text-right font-semibold">Median decision</th>
                  </tr>
                </thead>
                <tbody>
                  {summary.map((row) => {
                    const isAll = row.group === "ALL";
                    const label = isAll ? "All cases" : row.group;
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
                        <td className="px-3 py-2 text-right tabular-nums">
                          {row.mean_decision_seconds !== null
                            ? formatSeconds(row.mean_decision_seconds)
                            : "—"}
                        </td>
                        <td className="px-3 py-2 text-right tabular-nums">
                          {row.median_decision_seconds !== null
                            ? formatSeconds(row.median_decision_seconds)
                            : "—"}
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
                <span className="break-all text-slate-700">
                  <span className="font-mono font-semibold">
                    {entry.elapsed_hms}
                  </span>{" "}
                  · {entry.case_id ?? "—"}
                  {entry.pre_grade ? ` · pre ${entry.pre_grade}` : ""}
                  {entry.post_grade ? ` → post ${entry.post_grade}` : ""}
                  {typeof entry.decision_time_seconds === "number"
                    ? ` · decision ${formatSeconds(entry.decision_time_seconds)}`
                    : ""}
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
