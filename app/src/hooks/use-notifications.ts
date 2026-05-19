import { useCallback, useEffect, useRef } from "react"

import type { JobStatus, JobSummary } from "@/lib/types"
import { formatScore } from "@/lib/format"
import { statusText } from "@/lib/format"

/* ───────────────────────────────────────────
   useNotifications

   Watches dashboard jobs/selectedJob for
   running→completed/failed transitions and
   fires browser Notification API alerts.
   Deduplicates per job-id so each job
   notifies exactly once per page-session.

   Usage:
     const { requestPermission, isSupported, permission } =
       useNotifications({ jobs: dashboard.jobs, selectedJob: dashboard.selectedJob })

   Then in the workspace submit flow call
     await requestPermission()
   before submitting.
─────────────────────────────────────────── */

type LightJob = Pick<
  JobSummary,
  "id" | "status" | "run_name" | "average_score"
>

export function useNotifications({
  jobs,
  selectedJob,
}: {
  jobs: LightJob[]
  selectedJob: { id: string; status: JobStatus; run_name: string; average_score: number | null } | null
}) {
  /** IDs we have already notified. Resets on page reload. */
  const notifiedRef = useRef<Set<string>>(new Set())

  /** Previous status of every job we have seen. Used to detect transitions. */
  const prevStatusRef = useRef<Map<string, JobStatus>>(new Map())

  /* ── Permission request ────────────────── */
  const requestPermission = useCallback(async (): Promise<NotificationPermission> => {
    if (typeof window === "undefined" || !("Notification" in window)) {
      return "denied"
    }
    if (Notification.permission === "granted") {
      return "granted"
    }
    if (Notification.permission === "denied") {
      return "denied"
    }
    return Notification.requestPermission()
  }, [])

  /* ── Transition watcher ────────────────── */
  useEffect(() => {
    if (typeof window === "undefined" || !("Notification" in window)) {
      return
    }
    if (Notification.permission !== "granted") {
      return
    }

    const prevStatuses = prevStatusRef.current
    const notified = notifiedRef.current

    /**
     * Check a single job record. Only notifies when the job
     * transitioned from "running" → "completed"/"failed" AND
     * we haven't already notified for this job-id.
     */
    function checkJob(job: LightJob) {
      const prev = prevStatuses.get(job.id)
      // Always update the stored status so the next render sees it.
      prevStatuses.set(job.id, job.status)

      if (prev !== "running") {
        return
      }
      if (job.status !== "completed" && job.status !== "failed") {
        return
      }
      if (notified.has(job.id)) {
        return
      }

      notified.add(job.id)

      const label = statusText(job.status as JobStatus)
      const score =
        job.average_score != null ? ` · 得分 ${formatScore(job.average_score)}` : ""
      const body = `任务 "${job.run_name}" ${label}${score}`

      try {
        const notification = new Notification("PV-IQA 任务通知", {
          body,
          icon: "/favicon.ico",
          tag: job.id, // browser-level dedup by tag
        })
        // Auto-dismiss after 6 s so it doesn't linger.
        setTimeout(() => {
          try {
            notification.close()
          } catch {
            /* ignore */
          }
        }, 6000)
      } catch {
        // Permission may have been revoked between the check and the constructor.
      }
    }

    // Walk both the list and the detail record – dedup is handled by the Set.
    for (const job of jobs) {
      checkJob(job)
    }
    if (selectedJob) {
      checkJob(selectedJob)
    }
  }, [jobs, selectedJob])

  /* ── Derived values (re-evaluated every render on purpose) ── */
  const isSupported =
    typeof window !== "undefined" && "Notification" in window
  const permission: NotificationPermission =
    typeof window !== "undefined" && "Notification" in window
      ? Notification.permission
      : "denied"

  return {
    requestPermission,
    isSupported,
    permission,
  } as const
}
