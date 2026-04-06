export type JobKind = "image" | "folder"
export type JobStatus = "running" | "completed" | "failed"

export type UploadItem = {
  file: File
  relativePath: string
}

export type ScoreResult = {
  id: number
  job_id: string
  image_path: string
  relative_path: string
  public_url: string
  quality_score: number
}

export type JobSummary = {
  id: string
  kind: JobKind
  status: JobStatus
  run_name: string
  input_count: number
  processed_count: number
  result_count: number
  progress: number
  stage: string
  created_at: string
  updated_at: string
  completed_at: string | null
  error: string | null
}

export type JobRecord = JobSummary & {
  results: ScoreResult[]
}

export type HealthResponse = {
  status: "ok" | "degraded"
  port: number
  defaultRunName?: string
  error?: string
}
