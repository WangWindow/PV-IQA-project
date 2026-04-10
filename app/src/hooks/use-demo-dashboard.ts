import { useCallback, useEffect, useMemo, useState } from "react"

import {
  deleteJob,
  fetchHealth,
  fetchJob,
  fetchJobs,
  rerunJob as rerunJobRequest,
  resumeJob as resumeJobRequest,
  stopJob as stopJobRequest,
  submitFolder,
  submitSingleImage,
} from "@/lib/api"
import type {
  HealthResponse,
  InferenceBackend,
  JobRecord,
  JobSummary,
  UploadItem,
} from "@/lib/types"
import { averageScore } from "@/lib/demo-format"
import { filesToUploadItems, readDroppedItems } from "@/lib/uploads"

export type UploadMode = "image" | "folder"

export type DemoDashboardState = {
  mode: UploadMode
  health: HealthResponse | null
  jobs: JobSummary[]
  selectedJob: JobRecord | null
  backend: InferenceBackend
  runName: string
  singleFile: File | null
  folderItems: UploadItem[]
  isLoading: boolean
  isSubmitting: boolean
  isDragging: boolean
  deletingJobId: string | null
  mutatingJobId: string | null
  error: string | null
  selectedResults: JobRecord["results"]
  topResult: JobRecord["results"][number] | null
  summaryScore: number | null
  activeRunningCount: number
  completedCount: number
  failedCount: number
}

export type DemoDashboardActions = {
  setMode: (mode: UploadMode) => void
  setBackend: (backend: InferenceBackend) => void
  setRunName: (value: string) => void
  setDragging: (value: boolean) => void
  clearError: () => void
  resetUploads: () => void
  selectImageFile: (file: File | null) => void
  selectFolderFiles: (files: Iterable<File>) => void
  handleDrop: (dataTransfer: DataTransfer) => Promise<void>
  submit: () => Promise<void>
  selectJob: (jobId: string) => Promise<void>
  stopJob: (jobId: string) => Promise<void>
  resumeJob: (jobId: string) => Promise<void>
  rerunJob: (jobId: string) => Promise<void>
  removeJob: (jobId: string) => Promise<void>
}

export type DemoDashboard = DemoDashboardState & DemoDashboardActions

type DemoDashboardPreferences = {
  backend?: InferenceBackend
  runName?: string
}

const DASHBOARD_PREFERENCES_KEY = "pv-iqa-demo-preferences:v1"

function readDashboardPreferences(): DemoDashboardPreferences {
  if (typeof window === "undefined") {
    return {}
  }

  try {
    const raw = window.localStorage.getItem(DASHBOARD_PREFERENCES_KEY)
    if (!raw) {
      return {}
    }

    const parsed = JSON.parse(raw) as DemoDashboardPreferences
    return {
      backend: parsed.backend === "python" || parsed.backend === "rust" ? parsed.backend : undefined,
      runName: typeof parsed.runName === "string" ? parsed.runName : undefined,
    }
  } catch {
    return {}
  }
}

function writeDashboardPreferences(preferences: {
  backend: InferenceBackend
  runName: string
}): void {
  if (typeof window === "undefined") {
    return
  }

  try {
    window.localStorage.setItem(DASHBOARD_PREFERENCES_KEY, JSON.stringify(preferences))
  } catch {
    // Ignore transient storage failures and keep the in-memory state.
  }
}

export function useDemoDashboard(): DemoDashboard {
  const [storedPreferences] = useState<DemoDashboardPreferences>(() => readDashboardPreferences())
  const [mode, setMode] = useState<UploadMode>("image")
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [jobs, setJobs] = useState<JobSummary[]>([])
  const [selectedJob, setSelectedJob] = useState<JobRecord | null>(null)
  const [backend, setBackend] = useState<InferenceBackend>(storedPreferences.backend ?? "python")
  const [runName, setRunName] = useState(storedPreferences.runName ?? "")
  const [singleFile, setSingleFile] = useState<File | null>(null)
  const [folderItems, setFolderItems] = useState<UploadItem[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [isDragging, setIsDragging] = useState(false)
  const [deletingJobId, setDeletingJobId] = useState<string | null>(null)
  const [mutatingJobId, setMutatingJobId] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const refreshHealth = useCallback(async () => {
    const nextHealth = await fetchHealth()
    setHealth(nextHealth)
    return nextHealth
  }, [])

  const refreshJobs = useCallback(async () => {
    const nextJobs = await fetchJobs()
    setJobs(nextJobs)
    return nextJobs
  }, [])

  useEffect(() => {
    let disposed = false

    async function bootstrap() {
      try {
        const [, nextJobs] = await Promise.all([refreshHealth(), fetchJobs()])
        if (disposed) {
          return
        }

        setJobs(nextJobs)

        if (nextJobs[0]) {
          const initialJob = await fetchJob(nextJobs[0].id)
          if (!disposed) {
            setSelectedJob(initialJob)
          }
        }
      } catch (caughtError) {
        if (!disposed) {
          setError(String(caughtError))
        }
      } finally {
        if (!disposed) {
          setIsLoading(false)
        }
      }
    }

    void bootstrap()

    return () => {
      disposed = true
    }
  }, [refreshHealth])

  useEffect(() => {
    if (isLoading) {
      return
    }

    let disposed = false
    const intervalMs = health?.backends.rust.state === "starting" ? 2000 : 10000

    async function pollHealth() {
      try {
        const nextHealth = await fetchHealth()
        if (!disposed) {
          setHealth(nextHealth)
        }
      } catch {
        // Keep the last known health on transient polling failures.
      }
    }

    const timer = window.setInterval(() => {
      void pollHealth()
    }, intervalMs)

    return () => {
      disposed = true
      window.clearInterval(timer)
    }
  }, [health?.backends.rust.state, isLoading])

  const selectedJobId = selectedJob?.id ?? null
  const hasRunningJob =
    selectedJob?.status === "running" || jobs.some((job) => job.status === "running")

  useEffect(() => {
    if (!hasRunningJob) {
      return
    }

    let disposed = false

    async function pollJobs() {
      try {
        const [nextJobs, nextSelected] = await Promise.all([
          fetchJobs(),
          selectedJobId ? fetchJob(selectedJobId).catch(() => null) : Promise.resolve(null),
        ])

        if (disposed) {
          return
        }

        setJobs(nextJobs)
        if (nextSelected) {
          setSelectedJob(nextSelected)
        } else if (selectedJobId && !nextJobs.some((job) => job.id === selectedJobId)) {
          setSelectedJob(null)
        }
      } catch (caughtError) {
        if (!disposed) {
          setError(String(caughtError))
        }
      }
    }

    void pollJobs()
    const timer = window.setInterval(() => {
      void pollJobs()
    }, 1500)

    return () => {
      disposed = true
      window.clearInterval(timer)
    }
  }, [hasRunningJob, selectedJobId])

  useEffect(() => {
    if (backend !== "rust") {
      return
    }
    void refreshHealth().catch(() => undefined)
  }, [backend, refreshHealth])

  useEffect(() => {
    writeDashboardPreferences({ backend, runName })
  }, [backend, runName])

  const selectedResults = useMemo(
    () => [...(selectedJob?.results ?? [])].sort((left, right) => right.quality_score - left.quality_score),
    [selectedJob]
  )
  const topResult = selectedResults[0] ?? null
  const summaryScore = averageScore(selectedResults)
  const activeRunningCount = jobs.filter((job) => job.status === "running").length
  const completedCount = jobs.filter((job) => job.status === "completed").length
  const failedCount = jobs.filter((job) => job.status === "failed").length

  const clearError = useCallback(() => {
    setError(null)
  }, [])

  const resetUploads = useCallback(() => {
    setSingleFile(null)
    setFolderItems([])
  }, [])

  const selectImageFile = useCallback((file: File | null) => {
    setSingleFile(file)
  }, [])

  const selectFolderFiles = useCallback((files: Iterable<File>) => {
    setFolderItems(filesToUploadItems(files, "folder"))
  }, [])

  const handleDrop = useCallback(
    async (dataTransfer: DataTransfer) => {
      const droppedItems = await readDroppedItems(dataTransfer, mode)
      if (mode === "image") {
        setSingleFile(droppedItems[0]?.file ?? null)
        return
      }
      setFolderItems(droppedItems)
    },
    [mode]
  )

  const submit = useCallback(async () => {
    try {
      setError(null)
      setIsSubmitting(true)
      const effectiveRunName = runName.trim() || health?.defaultRunName
      let nextJob: JobRecord

        if (mode === "image") {
          if (!singleFile) {
            throw new Error("请先选择一张图片。")
          }
          nextJob = await submitSingleImage(singleFile, backend, effectiveRunName)
        } else {
          if (!folderItems.length) {
            throw new Error("请先选择一个文件夹。")
          }
          nextJob = await submitFolder(folderItems, backend, effectiveRunName)
        }

      setSelectedJob(nextJob)
      await refreshJobs()
      resetUploads()
    } catch (caughtError) {
      setError(String(caughtError))
    } finally {
      setIsSubmitting(false)
    }
  }, [backend, folderItems, health?.defaultRunName, mode, refreshJobs, resetUploads, runName, singleFile])

  const selectJob = useCallback(async (jobId: string) => {
    try {
      setError(null)
      setSelectedJob(await fetchJob(jobId))
    } catch (caughtError) {
      setError(String(caughtError))
    }
  }, [])

  const removeJob = useCallback(
    async (jobId: string) => {
      try {
        setError(null)
        setDeletingJobId(jobId)
        await deleteJob(jobId)
        const nextJobs = await refreshJobs()

        if (selectedJob?.id === jobId) {
          if (!nextJobs[0]) {
            setSelectedJob(null)
          } else {
            setSelectedJob(await fetchJob(nextJobs[0].id))
          }
        }
      } catch (caughtError) {
        setError(String(caughtError))
      } finally {
        setDeletingJobId(null)
      }
    },
    [refreshJobs, selectedJob?.id]
  )

  const stopJob = useCallback(
    async (jobId: string) => {
      try {
        setError(null)
        setMutatingJobId(jobId)
        const nextJob = await stopJobRequest(jobId)
        setSelectedJob(nextJob)
        await refreshJobs()
      } catch (caughtError) {
        setError(String(caughtError))
      } finally {
        setMutatingJobId(null)
      }
    },
    [refreshJobs]
  )

  const resumeJob = useCallback(
    async (jobId: string) => {
      try {
        setError(null)
        setMutatingJobId(jobId)
        const nextJob = await resumeJobRequest(jobId)
        setSelectedJob(nextJob)
        await refreshJobs()
      } catch (caughtError) {
        setError(String(caughtError))
      } finally {
        setMutatingJobId(null)
      }
    },
    [refreshJobs]
  )

  const rerunJob = useCallback(
    async (jobId: string) => {
      try {
        setError(null)
        setMutatingJobId(jobId)
        const nextJob = await rerunJobRequest(jobId)
        setSelectedJob(nextJob)
        await refreshJobs()
      } catch (caughtError) {
        setError(String(caughtError))
      } finally {
        setMutatingJobId(null)
      }
    },
    [refreshJobs]
  )

  return {
    mode,
    health,
    jobs,
    selectedJob,
    backend,
    runName,
    singleFile,
    folderItems,
    isLoading,
    isSubmitting,
    isDragging,
    deletingJobId,
    mutatingJobId,
    error,
    selectedResults,
    topResult,
    summaryScore,
    activeRunningCount,
    completedCount,
    failedCount,
    setMode,
    setBackend,
    setRunName,
    setDragging: setIsDragging,
    clearError,
    resetUploads,
    selectImageFile,
    selectFolderFiles,
    handleDrop,
    submit,
    selectJob,
    stopJob,
    resumeJob,
    rerunJob,
    removeJob,
  }
}
