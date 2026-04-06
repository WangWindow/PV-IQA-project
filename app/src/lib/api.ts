import type { HealthResponse, JobRecord, JobSummary, UploadItem } from "@/lib/types"

async function unwrapJson<T>(response: Response): Promise<T> {
  const payload = (await response.json()) as T & { error?: string }
  if (!response.ok) {
    throw new Error(payload.error || "Request failed.")
  }
  return payload
}

export async function fetchHealth(): Promise<HealthResponse> {
  return unwrapJson<HealthResponse>(await fetch("/api/health"))
}

export async function fetchJobs(): Promise<JobSummary[]> {
  const payload = await unwrapJson<{ jobs: JobSummary[] }>(await fetch("/api/jobs"))
  return payload.jobs
}

export async function fetchJob(jobId: string): Promise<JobRecord> {
  const payload = await unwrapJson<{ job: JobRecord }>(await fetch(`/api/jobs/${jobId}`))
  return payload.job
}

export async function deleteJob(jobId: string): Promise<void> {
  await unwrapJson<{ ok: boolean }>(
    await fetch(`/api/jobs/${jobId}`, {
      method: "DELETE",
    })
  )
}

export async function submitSingleImage(file: File, runName?: string): Promise<JobRecord> {
  const formData = new FormData()
  formData.append("file", file)
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

export async function submitFolder(items: UploadItem[], runName?: string): Promise<JobRecord> {
  const formData = new FormData()
  const manifest = items.map((item) => ({
    name: item.file.name,
    relativePath: item.relativePath,
  }))

  for (const item of items) {
    formData.append("files", item.file)
  }
  formData.append("manifest", JSON.stringify(manifest))

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
