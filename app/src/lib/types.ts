export type JobKind = "image" | "folder"
export type JobStatus = "running" | "interrupted" | "completed" | "failed"
export type InferenceBackend = "python" | "rust"
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
  backend: InferenceBackend
  device: "cpu" | "cuda"
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
  average_score: number | null
  best_score: number | null
  worst_score: number | null
}

export type JobRecord = JobSummary & {
  results: ScoreResult[]
}

export type BackendHealth = {
  available: boolean
  label: string
  state?: "ready" | "starting" | "idle" | "error"
  detail?: string
  device?: string
  error?: string
}

export type HealthResponse = {
  status: "ok" | "degraded"
  port: number
  defaultRunName?: string
  error?: string
  backends: {
    python: BackendHealth
    rust: BackendHealth
  }
}
