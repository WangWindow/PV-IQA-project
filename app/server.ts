import { Database } from "bun:sqlite"
import { mkdir, readdir, rm } from "node:fs/promises"
import { basename, dirname, normalize, resolve } from "node:path"

type JobKind = "image" | "folder"
type JobStatus = "running" | "completed" | "failed"
type UploadedFile = Blob & { name: string }
type MultipartEntry = string | UploadedFile

type JobSummary = {
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

type ResultRecord = {
  id: number
  job_id: string
  image_path: string
  relative_path: string
  public_url: string
  quality_score: number
}

type JobRecord = JobSummary & {
  results: ResultRecord[]
}

type SavedUpload = {
  savedPath: string
  relativePath: string
  publicUrl: string
}

type JobState = {
  status: JobStatus
  processedCount: number
  progress: number
  stage: string
  completedAt: string | null
  error: string | null
}

type ImageDetectionPayload = {
  run_directory: string
  result: {
    image_path: string
    quality_score: number
  }
}

type FormDataLike = {
  get(name: string): unknown
}

const APP_ROOT = import.meta.dir
const REPO_ROOT = resolve(APP_ROOT, "..")
const DATA_ROOT = resolve(APP_ROOT, "data")
const UPLOAD_ROOT = resolve(DATA_ROOT, "uploads")
const DIST_ROOT = resolve(APP_ROOT, "dist")
const PORT = Number(process.env.PORT ?? (process.env.NODE_ENV === "production" ? 6006 : 6007))

await mkdir(DATA_ROOT, { recursive: true })
await mkdir(UPLOAD_ROOT, { recursive: true })

const db = new Database(resolve(DATA_ROOT, "demo.sqlite"), { create: true })
db.exec(`
  PRAGMA journal_mode = WAL;
  PRAGMA foreign_keys = ON;
  CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    run_name TEXT NOT NULL,
    input_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    completed_at TEXT,
    error TEXT
  );
  CREATE TABLE IF NOT EXISTS results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    image_path TEXT NOT NULL,
    relative_path TEXT NOT NULL,
    public_url TEXT NOT NULL,
    quality_score REAL NOT NULL,
    FOREIGN KEY(job_id) REFERENCES jobs(id) ON DELETE CASCADE
  );
`)

function ensureColumn(tableName: string, columnName: string, definition: string): void {
  const columns = db.query(`PRAGMA table_info(${tableName})`).all() as Array<{ name: string }>
  if (!columns.some((column) => column.name === columnName)) {
    db.exec(`ALTER TABLE ${tableName} ADD COLUMN ${definition};`)
  }
}

ensureColumn("jobs", "processed_count", "processed_count INTEGER NOT NULL DEFAULT 0")
ensureColumn("jobs", "progress", "progress REAL NOT NULL DEFAULT 0")
ensureColumn("jobs", "stage", "stage TEXT NOT NULL DEFAULT '等待中'")
ensureColumn("jobs", "updated_at", "updated_at TEXT NOT NULL DEFAULT ''")

db.exec(`
  UPDATE jobs
  SET processed_count = COALESCE(processed_count, 0),
      progress = COALESCE(progress, 0),
      stage = CASE WHEN stage IS NULL OR stage = '' THEN '等待中' ELSE stage END,
      updated_at = CASE WHEN updated_at IS NULL OR updated_at = '' THEN created_at ELSE updated_at END;
`)

const insertJobQuery = db.query(
  "INSERT INTO jobs (id, kind, status, run_name, input_count, processed_count, progress, stage, created_at, updated_at, completed_at, error) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)
const updateJobStateQuery = db.query(
  "UPDATE jobs SET status = ?, processed_count = ?, progress = ?, stage = ?, updated_at = ?, completed_at = ?, error = ? WHERE id = ?"
)
const insertResultQuery = db.query(
  "INSERT INTO results (job_id, image_path, relative_path, public_url, quality_score) VALUES (?, ?, ?, ?, ?)"
)
const deleteResultsQuery = db.query("DELETE FROM results WHERE job_id = ?")
const deleteJobQuery = db.query("DELETE FROM jobs WHERE id = ?")

function jsonResponse(payload: unknown, init: ResponseInit = {}): Response {
  return Response.json(payload, init)
}

function errorResponse(message: string, status = 400): Response {
  return jsonResponse({ error: message }, { status })
}

function nowIso(): string {
  return new Date().toISOString()
}

function sanitizeSegment(value: string): string {
  return value.replace(/[^a-zA-Z0-9._-]/g, "_")
}

function sanitizeRelativePath(rawPath: string): string {
  const normalized = normalize(rawPath.replace(/\\/g, "/")).replace(/^(\.\.(\/|\\|$))+/, "")
  const parts = normalized.split("/").filter(Boolean).map(sanitizeSegment)
  return parts.join("/") || sanitizeSegment(basename(rawPath))
}

function publicUploadUrl(jobId: string, relativePath: string): string {
  return `/uploads/${jobId}/${sanitizeRelativePath(relativePath)}`
}

function isUploadedFile(entry: MultipartEntry): entry is UploadedFile {
  return typeof entry !== "string"
}

async function saveUploadedFile(jobId: string, relativePath: string, file: UploadedFile): Promise<SavedUpload> {
  const cleanedPath = sanitizeRelativePath(relativePath)
  const targetPath = resolve(UPLOAD_ROOT, jobId, cleanedPath)
  await mkdir(dirname(targetPath), { recursive: true })
  await Bun.write(targetPath, file)
  return {
    savedPath: targetPath,
    relativePath: cleanedPath,
    publicUrl: publicUploadUrl(jobId, cleanedPath),
  }
}

async function resolveDefaultRunName(): Promise<string> {
  if (process.env.PV_IQA_RUN_NAME) {
    return process.env.PV_IQA_RUN_NAME
  }

  const checkpointsRoot = resolve(REPO_ROOT, "checkpoints")
  const entries = await readdir(checkpointsRoot, { withFileTypes: true }).catch(() => [])
  const candidates = await Promise.all(
    entries
      .filter((entry) => entry.isDirectory())
      .map(async (entry) => {
        const checkpointFile = Bun.file(resolve(checkpointsRoot, entry.name, "iqa", "best.pt"))
        if (!(await checkpointFile.exists())) {
          return null
        }
        return {
          name: entry.name,
          updatedAt: checkpointFile.lastModified,
        }
      })
  )

  const latest = candidates
    .filter((item): item is { name: string; updatedAt: number } => item !== null)
    .sort((left, right) => right.updatedAt - left.updatedAt)[0]

  if (!latest) {
    throw new Error("No completed checkpoint run found under checkpoints/.")
  }
  return latest.name
}

async function runPvIqa(args: string[]): Promise<unknown> {
  const command = Bun.spawn(["uv", "run", "pv-iqa", ...args], {
    cwd: REPO_ROOT,
    stdout: "pipe",
    stderr: "pipe",
    env: {
      ...process.env,
      PYTHONIOENCODING: "utf-8",
    },
  })

  const stdoutPromise = new Response(command.stdout).text()
  const stderrPromise = new Response(command.stderr).text()
  const exitCode = await command.exited
  const stdout = (await stdoutPromise).trim()
  const stderr = (await stderrPromise).trim()

  if (exitCode !== 0) {
    throw new Error(stderr || stdout || "pv-iqa command failed")
  }

  try {
    return JSON.parse(stdout)
  } catch {
    throw new Error(`Failed to parse pv-iqa JSON output: ${stdout}`)
  }
}

function clampProgress(value: number): number {
  return Math.max(0, Math.min(100, value))
}

function createJob(jobId: string, kind: JobKind, runName: string, inputCount: number, state: JobState): void {
  const timestamp = nowIso()
  insertJobQuery.run(
    jobId,
    kind,
    state.status,
    runName,
    inputCount,
    state.processedCount,
    clampProgress(state.progress),
    state.stage,
    timestamp,
    timestamp,
    state.completedAt,
    state.error
  )
}

function persistJobState(jobId: string, state: JobState): void {
  updateJobStateQuery.run(
    state.status,
    state.processedCount,
    clampProgress(state.progress),
    state.stage,
    nowIso(),
    state.completedAt,
    state.error,
    jobId
  )
}

function loadJobSummary(jobId: string): JobSummary | null {
  return db
    .query(
      `SELECT
         jobs.id,
         jobs.kind,
         jobs.status,
         jobs.run_name,
         jobs.input_count,
         jobs.processed_count,
         jobs.progress,
         jobs.stage,
         jobs.created_at,
         jobs.updated_at,
         jobs.completed_at,
         jobs.error,
         (SELECT COUNT(*) FROM results WHERE job_id = jobs.id) AS result_count
       FROM jobs
       WHERE jobs.id = ?`
    )
    .get(jobId) as JobSummary | null
}

function loadJob(jobId: string): JobRecord | null {
  const job = loadJobSummary(jobId)
  if (!job) {
    return null
  }

  const results = db
    .query(
      "SELECT id, job_id, image_path, relative_path, public_url, quality_score FROM results WHERE job_id = ? ORDER BY quality_score DESC, relative_path ASC"
    )
    .all(jobId) as ResultRecord[]

  return { ...job, results }
}

function listJobs(limit = 20): JobSummary[] {
  return db
    .query(
      `SELECT
         jobs.id,
         jobs.kind,
         jobs.status,
         jobs.run_name,
         jobs.input_count,
         jobs.processed_count,
         jobs.progress,
         jobs.stage,
         jobs.created_at,
         jobs.updated_at,
         jobs.completed_at,
         jobs.error,
         (SELECT COUNT(*) FROM results WHERE job_id = jobs.id) AS result_count
       FROM jobs
       ORDER BY jobs.created_at DESC
       LIMIT ?`
    )
    .all(limit) as JobSummary[]
}

async function detectImage(savedPath: string, runName: string): Promise<ImageDetectionPayload["result"]> {
  const output = (await runPvIqa([
    "detect-image",
    savedPath,
    "--run-name",
    runName,
    "--json-output",
  ])) as ImageDetectionPayload
  return output.result
}

async function processImageJob(
  jobId: string,
  runName: string,
  upload: SavedUpload
): Promise<void> {
  const state: JobState = {
    status: "running",
    processedCount: 0,
    progress: 25,
    stage: "模型评分中",
    completedAt: null,
    error: null,
  }
  persistJobState(jobId, state)

  try {
    const result = await detectImage(upload.savedPath, runName)
    insertResultQuery.run(
      jobId,
      result.image_path,
      upload.relativePath,
      upload.publicUrl,
      result.quality_score
    )

    state.status = "completed"
    state.processedCount = 1
    state.progress = 100
    state.stage = "评分完成"
    state.completedAt = nowIso()
    persistJobState(jobId, state)
  } catch (error) {
    state.status = "failed"
    state.progress = Math.max(state.progress, 25)
    state.stage = "评分失败"
    state.error = String(error)
    state.completedAt = nowIso()
    persistJobState(jobId, state)
  }
}

async function processFolderJob(
  jobId: string,
  runName: string,
  uploads: SavedUpload[]
): Promise<void> {
  const total = uploads.length
  const state: JobState = {
    status: "running",
    processedCount: 0,
    progress: total ? 10 : 100,
    stage: total ? "等待批量评分" : "没有可评分图片",
    completedAt: null,
    error: null,
  }
  persistJobState(jobId, state)

  try {
    for (const [index, upload] of uploads.entries()) {
      state.stage = `批量评分中 ${index + 1}/${total}`
      state.progress = 15 + (index / total) * 75
      persistJobState(jobId, state)

      const result = await detectImage(upload.savedPath, runName)
      insertResultQuery.run(
        jobId,
        result.image_path,
        upload.relativePath,
        upload.publicUrl,
        result.quality_score
      )

      state.processedCount = index + 1
      state.progress = 15 + ((index + 1) / total) * 75
      state.stage = `已完成 ${state.processedCount}/${total}`
      persistJobState(jobId, state)
    }

    state.status = "completed"
    state.progress = 100
    state.stage = "批量评分完成"
    state.completedAt = nowIso()
    persistJobState(jobId, state)
  } catch (error) {
    state.status = "failed"
    state.progress = Math.max(state.progress, 15)
    state.stage = "批量评分失败"
    state.error = String(error)
    state.completedAt = nowIso()
    persistJobState(jobId, state)
  }
}

async function parseRunName(formData: FormDataLike): Promise<string> {
  const rawRunName = String(formData.get("runName") || "").trim()
  return rawRunName || (await resolveDefaultRunName())
}

async function handleImageScore(request: Request): Promise<Response> {
  const formData = await request.formData()
  const file = formData.get("file") as MultipartEntry | null
  if (!file || !isUploadedFile(file)) {
    return errorResponse("Missing image file.")
  }

  try {
    const runName = await parseRunName(formData)
    const jobId = crypto.randomUUID()
    const upload = await saveUploadedFile(jobId, file.name, file)

    createJob(jobId, "image", runName, 1, {
      status: "running",
      processedCount: 0,
      progress: 10,
      stage: "上传完成，等待评分",
      completedAt: null,
      error: null,
    })

    void processImageJob(jobId, runName, upload)
    return jsonResponse({ job: loadJob(jobId) }, { status: 202 })
  } catch (error) {
    return errorResponse(String(error), 500)
  }
}

async function handleFolderScore(request: Request): Promise<Response> {
  const formData = await request.formData()
  const files = (formData.getAll("files") as MultipartEntry[]).filter(isUploadedFile)
  if (!files.length) {
    return errorResponse("No images were uploaded.")
  }

  try {
    const manifest = JSON.parse(String(formData.get("manifest") || "[]")) as Array<{
      name: string
      relativePath: string
    }>
    const runName = await parseRunName(formData)
    const jobId = crypto.randomUUID()
    const uploads: SavedUpload[] = []

    for (const [index, file] of files.entries()) {
      const relativePath = manifest[index]?.relativePath || file.name
      uploads.push(await saveUploadedFile(jobId, relativePath, file))
    }

    createJob(jobId, "folder", runName, uploads.length, {
      status: "running",
      processedCount: 0,
      progress: 8,
      stage: "上传完成，等待批量评分",
      completedAt: null,
      error: null,
    })

    void processFolderJob(jobId, runName, uploads)
    return jsonResponse({ job: loadJob(jobId) }, { status: 202 })
  } catch (error) {
    return errorResponse(String(error), 500)
  }
}

async function handleDeleteJob(jobId: string): Promise<Response> {
  const job = loadJobSummary(jobId)
  if (!job) {
    return errorResponse("Job not found.", 404)
  }
  if (job.status === "running") {
    return errorResponse("Running jobs cannot be deleted.", 409)
  }

  deleteResultsQuery.run(jobId)
  deleteJobQuery.run(jobId)
  await rm(resolve(UPLOAD_ROOT, jobId), { recursive: true, force: true }).catch(() => undefined)
  return jsonResponse({ ok: true })
}

async function serveUpload(pathname: string): Promise<Response> {
  const relativeUploadPath = pathname.replace(/^\/uploads\//, "")
  const localPath = resolve(UPLOAD_ROOT, relativeUploadPath)
  if (!localPath.startsWith(UPLOAD_ROOT)) {
    return errorResponse("Invalid upload path.", 403)
  }
  const file = Bun.file(localPath)
  if (!(await file.exists())) {
    return errorResponse("Upload not found.", 404)
  }
  return new Response(file)
}

async function serveDist(pathname: string): Promise<Response> {
  const requestedPath = pathname === "/" ? "/index.html" : pathname
  const localPath = resolve(DIST_ROOT, "." + requestedPath)
  const requestedFile = Bun.file(localPath)
  if (await requestedFile.exists()) {
    return new Response(requestedFile)
  }

  const indexFile = Bun.file(resolve(DIST_ROOT, "index.html"))
  if (await indexFile.exists()) {
    return new Response(indexFile)
  }
  return errorResponse("Frontend build not found.", 404)
}

const server = Bun.serve({
  port: PORT,
  async fetch(request) {
    const url = new URL(request.url)

    if (url.pathname === "/api/health") {
      try {
        return jsonResponse({
          status: "ok",
          defaultRunName: await resolveDefaultRunName(),
          port: PORT,
        })
      } catch (error) {
        return jsonResponse(
          {
            status: "degraded",
            error: String(error),
            port: PORT,
          },
          { status: 500 }
        )
      }
    }

    if (url.pathname === "/api/jobs" && request.method === "GET") {
      return jsonResponse({ jobs: listJobs() })
    }

    if (url.pathname.startsWith("/api/jobs/")) {
      const jobId = url.pathname.split("/").pop()
      if (!jobId) {
        return errorResponse("Job not found.", 404)
      }

      if (request.method === "GET") {
        const job = loadJob(jobId)
        return job ? jsonResponse({ job }) : errorResponse("Job not found.", 404)
      }

      if (request.method === "DELETE") {
        return handleDeleteJob(jobId)
      }
    }

    if (url.pathname === "/api/score/image" && request.method === "POST") {
      return handleImageScore(request)
    }

    if (url.pathname === "/api/score/folder" && request.method === "POST") {
      return handleFolderScore(request)
    }

    if (url.pathname.startsWith("/uploads/")) {
      return serveUpload(url.pathname)
    }

    if (process.env.NODE_ENV === "production") {
      return serveDist(url.pathname)
    }

    return errorResponse("Not found.", 404)
  },
})

console.log(`PV-IQA demo bridge running on http://localhost:${server.port}`)
