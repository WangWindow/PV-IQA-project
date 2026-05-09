import type { HealthResponse, InferenceBackend, JobRecord, JobSummary, UploadItem } from "@/lib/types"

async function unwrapJson<T>(response: Response): Promise<T> {
  const payload = (await response.json()) as T & { error?: string }
  if (!response.ok) {
    throw new Error(payload.error || "Request failed.")
  }
  return payload
}

export async function fetchHealth(): Promise<HealthResponse> {
  return unwrapJson<HealthResponse>(await fetch("/api/health", { cache: "no-store" }))
}

export async function fetchJobs(): Promise<JobSummary[]> {
  const payload = await unwrapJson<{ jobs: JobSummary[] }>(
    await fetch("/api/jobs", { cache: "no-store" })
  )
  return payload.jobs
}

export async function fetchJob(jobId: string): Promise<JobRecord> {
  const payload = await unwrapJson<{ job: JobRecord }>(
    await fetch(`/api/jobs/${jobId}`, { cache: "no-store" })
  )
  return payload.job
}

export async function deleteJob(jobId: string): Promise<void> {
  await unwrapJson<{ ok: boolean }>(
    await fetch(`/api/jobs/${jobId}`, {
      method: "DELETE",
    })
  )
}

export async function stopJob(jobId: string): Promise<JobRecord> {
  const payload = await unwrapJson<{ job: JobRecord }>(
    await fetch(`/api/jobs/${jobId}/stop`, {
      method: "POST",
    })
  )
  return payload.job
}

export async function resumeJob(jobId: string): Promise<JobRecord> {
  const payload = await unwrapJson<{ job: JobRecord }>(
    await fetch(`/api/jobs/${jobId}/resume`, {
      method: "POST",
    })
  )
  return payload.job
}

export async function rerunJob(jobId: string): Promise<JobRecord> {
  const payload = await unwrapJson<{ job: JobRecord }>(
    await fetch(`/api/jobs/${jobId}/rerun`, {
      method: "POST",
    })
  )
  return payload.job
}

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
    await fetch("/api/score/image", {
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
    await fetch("/api/score/folder", {
      method: "POST",
      body: formData,
    })
  )
  return payload.job
}
