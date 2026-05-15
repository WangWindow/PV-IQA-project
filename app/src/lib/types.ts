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
  priority?: number
  tags?: string[]
  notes?: string
  user_id?: string | null
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

// ── 认证相关类型 ────────────────────────────────────────

export type UserInfo = {
  id: string
  username: string
  role: "admin" | "user"
  created_at?: string
  updated_at?: string
  job_count?: number
}

export type AuthState = {
  user: UserInfo | null
  token: string | null
  isAuthenticated: boolean
  isLoading: boolean
}

export type LoginRequest = {
  username: string
  password: string
}

export type RegisterRequest = {
  username: string
  password: string
}

export type AuthResponse = {
  user: UserInfo
  access_token: string
  token_type: string
}

// ── 图片元数据类型 ──────────────────────────────────────

export type ImageMetadata = {
  filename: string
  format: string | null
  size_bytes: number
  size_human: string
  width: number
  height: number
  mode: string | null
  dpi: [number, number] | null
  brightness: number | null
  contrast: number | null
  snr_estimate: number | null
  histogram: {
    r: number[]
    g: number[]
    b: number[]
    luminance: number[]
  } | null
}

// ── 审计日志类型 ────────────────────────────────────────

export type AuditLog = {
  id: number
  user_id: string | null
  action: string
  target_type: string | null
  target_id: string | null
  detail: string | null
  ip_address: string | null
  created_at: string
}

export type AuditLogPage = {
  logs: AuditLog[]
  pagination: {
    page: number
    page_size: number
    total: number
    total_pages: number
  }
}

// ── 系统设置类型 ────────────────────────────────────────

export type SystemSettings = {
  default_backend: string
  default_device: string
  max_upload_size_mb: string
  allowed_image_types: string
  job_retention_days: string
  system_name: string
}

// ── API 错误响应类型 ────────────────────────────────────

export type ApiError = {
  code: string
  message: string
  detail?: string | string[]
}