import type {
  AuditLog,
  AuditLogPage,
  AuthResponse,
  HealthResponse,
  ImageMetadata,
  InferenceBackend,
  JobRecord,
  JobSummary,
  LoginRequest,
  RegisterRequest,
  SystemSettings,
  UploadItem,
  UserInfo,
} from "@/lib/types"

// ── 通用请求封装 ────────────────────────────────────────

/** 获取存储的 JWT 令牌 */
function getAuthHeaders(): HeadersInit {
  const token = typeof window !== "undefined" ? localStorage.getItem("pv-iqa-token") : null
  return token ? { Authorization: `Bearer ${token}` } : {}
}

/** 解包 API 响应，处理结构化错误 */
async function unwrapJson<T>(response: Response): Promise<T> {
  const payload = (await response.json()) as T & { error?: string; code?: string; message?: string }

  if (!response.ok) {
    // 优先使用新的结构化错误格式
    const message = payload.message || payload.error || "请求失败"
    const code = payload.code || "UNKNOWN_ERROR"
    throw new Error(`[${code}] ${message}`)
  }

  return payload
}

/** 带认证的 fetch */
async function authFetch(url: string, options: RequestInit = {}): Promise<Response> {
  const headers = { ...getAuthHeaders(), ...options.headers }
  return fetch(url, { ...options, headers })
}

// ── 健康检查 ────────────────────────────────────────────

export async function fetchHealth(): Promise<HealthResponse> {
  return unwrapJson<HealthResponse>(await fetch("/api/health", { cache: "no-store" }))
}

// ── 任务管理 ────────────────────────────────────────────

export async function fetchRuns(): Promise<string[]> {
  return unwrapJson<string[]>(await fetch("/api/runs"))
}

export async function fetchJobs(): Promise<JobSummary[]> {
  const payload = await unwrapJson<{ jobs: JobSummary[] }>(
    await authFetch("/api/jobs", { cache: "no-store" })
  )
  return payload.jobs
}

export async function fetchJob(jobId: string): Promise<JobRecord> {
  const payload = await unwrapJson<{ job: JobRecord }>(
    await authFetch(`/api/jobs/${jobId}`, { cache: "no-store" })
  )
  return payload.job
}

export async function deleteJob(jobId: string): Promise<void> {
  await unwrapJson<{ ok: boolean }>(
    await authFetch(`/api/jobs/${jobId}`, { method: "DELETE" })
  )
}

export async function stopJob(jobId: string): Promise<JobRecord> {
  const payload = await unwrapJson<{ job: JobRecord }>(
    await authFetch(`/api/jobs/${jobId}/stop`, { method: "POST" })
  )
  return payload.job
}

export async function resumeJob(jobId: string): Promise<JobRecord> {
  const payload = await unwrapJson<{ job: JobRecord }>(
    await authFetch(`/api/jobs/${jobId}/resume`, { method: "POST" })
  )
  return payload.job
}

export async function rerunJob(jobId: string): Promise<JobRecord> {
  const payload = await unwrapJson<{ job: JobRecord }>(
    await authFetch(`/api/jobs/${jobId}/rerun`, { method: "POST" })
  )
  return payload.job
}

export async function batchDeleteJobs(jobIds: string[]): Promise<{ deleted: number }> {
  const payload = await unwrapJson<{ ok: boolean; deleted: number }>(
    await authFetch("/api/jobs/batch-delete", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ job_ids: jobIds }),
    })
  )
  return { deleted: payload.deleted }
}

export async function cleanupJobs(days: number): Promise<{ cleaned: number }> {
  const payload = await unwrapJson<{ ok: boolean; cleaned: number }>(
    await authFetch(`/api/jobs/cleanup?days=${days}`, { method: "POST" })
  )
  return { cleaned: payload.cleaned }
}

// ── 评分提交 ────────────────────────────────────────────

export async function submitSingleImage(
  file: File,
  backend: InferenceBackend,
  device: "cpu" | "cuda" = "cpu",
  runName?: string
): Promise<JobRecord> {
  const formData = new FormData()
  formData.append("file", file)
  formData.append("backend", backend)
  formData.append("device", device)
  if (runName) {
    formData.append("runName", runName)
  }
  const payload = await unwrapJson<{ job: JobRecord }>(
    await authFetch("/api/score/image", {
      method: "POST",
      body: formData,
    })
  )
  return payload.job
}

export async function submitFolder(
  items: UploadItem[],
  backend: InferenceBackend,
  device: "cpu" | "cuda" = "cpu",
  runName?: string
): Promise<JobRecord> {
  const formData = new FormData()
  const manifest = items.map((item) => ({
    name: item.file.name,
    relativePath: item.relativePath,
  }))

  for (const item of items) {
    formData.append("files", item.file)
  }
  formData.append("manifest", JSON.stringify(manifest))
  formData.append("backend", backend)
  formData.append("device", device)

  if (runName) {
    formData.append("runName", runName)
  }

  const payload = await unwrapJson<{ job: JobRecord }>(
    await authFetch("/api/score/folder", {
      method: "POST",
      body: formData,
    })
  )
  return payload.job
}

// ── 认证 ────────────────────────────────────────────────

export async function loginApi(data: LoginRequest): Promise<AuthResponse> {
  const payload = await unwrapJson<AuthResponse>(
    await fetch("/api/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    })
  )
  return payload
}

export async function registerApi(data: RegisterRequest): Promise<AuthResponse> {
  const payload = await unwrapJson<AuthResponse>(
    await fetch("/api/auth/register", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    })
  )
  return payload
}

export async function fetchCurrentUser(): Promise<UserInfo | null> {
  try {
    const payload = await unwrapJson<{ user: UserInfo }>(
      await authFetch("/api/auth/me")
    )
    return payload.user
  } catch {
    return null
  }
}

export async function changePassword(oldPassword: string, newPassword: string): Promise<void> {
  await unwrapJson<{ ok: boolean }>(
    await authFetch("/api/auth/password", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ old_password: oldPassword, new_password: newPassword }),
    })
  )
}

export async function listUsers(): Promise<UserInfo[]> {
  const payload = await unwrapJson<{ users: (UserInfo & { job_count?: number })[] }>(
    await authFetch("/api/auth/users")
  )
  return payload.users
}

export async function deleteUser(userId: string): Promise<void> {
  await unwrapJson<{ ok: boolean }>(
    await authFetch(`/api/auth/users/${userId}`, { method: "DELETE" })
  )
}

export async function updateUserRole(userId: string, role: "admin" | "user"): Promise<void> {
  await unwrapJson<{ ok: boolean }>(
    await authFetch(`/api/auth/users/${userId}/role`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ role }),
    })
  )
}

export async function adminCreateUser(username: string, password: string, role: string = "user"): Promise<UserInfo> {
  const payload = await unwrapJson<{ user: UserInfo }>(
    await authFetch("/api/auth/users/create", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password, role }),
    })
  )
  return payload.user
}

// ── 图片元数据 ──────────────────────────────────────────

export async function fetchImageMetadata(path: string): Promise<ImageMetadata> {
  const payload = await unwrapJson<{ metadata: ImageMetadata }>(
    await authFetch(`/api/images/metadata?path=${encodeURIComponent(path)}`)
  )
  return payload.metadata
}

// ── 审计日志 ────────────────────────────────────────────

export async function fetchLogs(params?: {
  userId?: string
  action?: string
  targetType?: string
  startDate?: string
  endDate?: string
  page?: number
  pageSize?: number
}): Promise<AuditLogPage> {
  const searchParams = new URLSearchParams()
  if (params?.userId) searchParams.set("user_id", params.userId)
  if (params?.action) searchParams.set("action", params.action)
  if (params?.targetType) searchParams.set("target_type", params.targetType)
  if (params?.startDate) searchParams.set("start_date", params.startDate)
  if (params?.endDate) searchParams.set("end_date", params.endDate)
  if (params?.page) searchParams.set("page", String(params.page))
  if (params?.pageSize) searchParams.set("page_size", String(params.pageSize))

  const query = searchParams.toString()
  const url = `/api/logs${query ? `?${query}` : ""}`
  return unwrapJson<AuditLogPage>(await authFetch(url))
}

export async function exportLogs(): Promise<AuditLog[]> {
  const payload = await unwrapJson<{ logs: AuditLog[] }>(
    await authFetch("/api/logs/export")
  )
  return payload.logs
}

// ── 系统设置 ────────────────────────────────────────────

export async function fetchSettings(): Promise<SystemSettings> {
  const payload = await unwrapJson<{ settings: SystemSettings }>(
    await authFetch("/api/settings")
  )
  return payload.settings
}

export async function updateSettings(settings: Partial<SystemSettings>): Promise<{ updated: string[] }> {
  const entries = Object.entries(settings).map(([key, value]) => ({ key, value }))
  const payload = await unwrapJson<{ ok: boolean; updated: string[] }>(
    await authFetch("/api/settings", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ settings: entries }),
    })
  )
  return { updated: payload.updated }
}

export async function fetchDbStats(): Promise<{ stats: { jobs: number; results: number; users: number; audit_logs: number } }> {
  return unwrapJson(await authFetch("/api/settings/db-stats"))
}