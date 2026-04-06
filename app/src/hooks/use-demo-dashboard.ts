import { useCallback, useEffect, useMemo, useState } from "react"

import {
  deleteJob,
  fetchHealth,
  fetchJob,
  fetchJobs,
  submitFolder,
  submitSingleImage,
} from "@/lib/api"
import type { HealthResponse, JobRecord, JobSummary, UploadItem } from "@/lib/types"
import { averageScore } from "@/lib/demo-format"
import { filesToUploadItems, readDroppedItems } from "@/lib/uploads"

export type UploadMode = "image" | "folder"

export type DemoDashboardState = {
  mode: UploadMode
  health: HealthResponse | null
  jobs: JobSummary[]
  selectedJob: JobRecord | null
  runName: string
  singleFile: File | null
  folderItems: UploadItem[]
  isLoading: boolean
  isSubmitting: boolean
  isDragging: boolean
  deletingJobId: string | null
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
  setRunName: (value: string) => void
  setDragging: (value: boolean) => void
  clearError: () => void
  resetUploads: () => void
  selectImageFile: (file: File | null) => void
  selectFolderFiles: (files: Iterable<File>) => void
  handleDrop: (dataTransfer: DataTransfer) => Promise<void>
  submit: () => Promise<void>
  selectJob: (jobId: string) => Promise<void>
  removeJob: (jobId: string) => Promise<void>
}

export type DemoDashboard = DemoDashboardState & DemoDashboardActions

export function useDemoDashboard(): DemoDashboard {
  const [mode, setMode] = useState<UploadMode>("image")
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [jobs, setJobs] = useState<JobSummary[]>([])
  const [selectedJob, setSelectedJob] = useState<JobRecord | null>(null)
  const [runName, setRunName] = useState("")
  const [singleFile, setSingleFile] = useState<File | null>(null)
  const [folderItems, setFolderItems] = useState<UploadItem[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [isDragging, setIsDragging] = useState(false)
  const [deletingJobId, setDeletingJobId] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const refreshJobs = useCallback(async () => {
    const nextJobs = await fetchJobs()
    setJobs(nextJobs)
    return nextJobs
  }, [])

  useEffect(() => {
    let disposed = false

    async function bootstrap() {
      try {
        const [nextHealth, nextJobs] = await Promise.all([fetchHealth(), fetchJobs()])
        if (disposed) {
          return
        }

        setHealth(nextHealth)
        setJobs(nextJobs)
        setRunName(nextHealth.defaultRunName ?? "")

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
  }, [])

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
        nextJob = await submitSingleImage(singleFile, effectiveRunName)
      } else {
        if (!folderItems.length) {
          throw new Error("请先选择一个文件夹。")
        }
        nextJob = await submitFolder(folderItems, effectiveRunName)
      }

      setSelectedJob(nextJob)
      await refreshJobs()
      resetUploads()
    } catch (caughtError) {
      setError(String(caughtError))
    } finally {
      setIsSubmitting(false)
    }
  }, [folderItems, health?.defaultRunName, mode, refreshJobs, resetUploads, runName, singleFile])

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

  return {
    mode,
    health,
    jobs,
    selectedJob,
    runName,
    singleFile,
    folderItems,
    isLoading,
    isSubmitting,
    isDragging,
    deletingJobId,
    error,
    selectedResults,
    topResult,
    summaryScore,
    activeRunningCount,
    completedCount,
    failedCount,
    setMode,
    setRunName,
    setDragging: setIsDragging,
    clearError,
    resetUploads,
    selectImageFile,
    selectFolderFiles,
    handleDrop,
    submit,
    selectJob,
    removeJob,
  }
}
