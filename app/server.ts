import { Database } from "bun:sqlite"
import { chmod, copyFile, mkdir, readdir, rm } from "node:fs/promises"
import { basename, dirname, normalize, resolve } from "node:path"

type JobKind = "image" | "folder"
type JobStatus = "running" | "interrupted" | "completed" | "failed"
type InferenceBackend = "python" | "rust"
type UploadedFile = Blob & { name: string }
type MultipartEntry = string | UploadedFile

type JobSummary = {
  id: string
  kind: JobKind
  status: JobStatus
  backend: InferenceBackend
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

type ActiveJobContext = {
  jobId: string
  kind: JobKind
  backend: InferenceBackend
  runName: string
  abortController: AbortController
  currentPythonProcess: ReturnType<typeof Bun.spawn> | null
  stopRequested: boolean
  stopReason: string | null
  done: Promise<void> | null
}

type ImageDetectionPayload = {
  run_directory: string
  result: {
    image_path: string
    quality_score: number
  }
}

type RustHealthPayload = {
  status: "ok" | "degraded"
  service: string
  device: string
  started_at: string
  cache_size: number
  loaded_runs: string[]
  repo_root: string
}

type PythonHealthPayload = {
  device: string
  runtime_device: string
  cuda_available: boolean
  gpu_name?: string | null
}

type RustImageDetectionPayload = {
  backend: string
  request_id: string
  run_name: string
  duration_ms: number
  result: {
    image_path: string
    quality_score: number
  }
}

type RustBatchDetectionPayload = {
  backend: string
  request_id: string
  run_name: string
  duration_ms: number
  result_count: number
  results: Array<{
    image_path: string
    quality_score: number
  }>
}

type FormDataLike = {
  get(name: string): unknown
}

const APP_ROOT = import.meta.dir
const REPO_ROOT = resolve(APP_ROOT, "..")
const BIN_ROOT = resolve(REPO_ROOT, "bin")
const DATA_ROOT = resolve(APP_ROOT, "data")
const UPLOAD_ROOT = resolve(DATA_ROOT, "uploads")
const DIST_ROOT = resolve(APP_ROOT, "dist")
const DEFAULT_PORT = process.env.NODE_ENV === "production" ? 6006 : 6005
const PORT = Number(process.env.PV_IQA_API_PORT ?? process.env.PORT ?? DEFAULT_PORT)
const RUST_SERVICE_URL = process.env.PV_IQA_RS_URL ?? "http://127.0.0.1:7007"
const RUST_SERVICE_ENDPOINT = new URL(RUST_SERVICE_URL)
const RUST_SERVICE_HOST = process.env.PV_IQA_RS_HOST ?? RUST_SERVICE_ENDPOINT.hostname
const RUST_SERVICE_PORT = Number(
  process.env.PV_IQA_RS_PORT ||
    RUST_SERVICE_ENDPOINT.port ||
    (RUST_SERVICE_ENDPOINT.protocol === "https:" ? "443" : "80")
)
const RUST_PROJECT_ROOT = process.env.PV_IQA_RS_PROJECT_ROOT ?? "/root/autodl-tmp/PV-IQA-rs"
const RUST_EXPORT_PROFILE = "candle-upstream-static-v2"
const RUST_CUDA_FEATURES = (process.env.PV_IQA_RS_CARGO_FEATURES ?? "cuda").trim() || "cuda"
const RUST_AUTOSTART = process.env.PV_IQA_RS_AUTOSTART !== "0"
const RUST_USE_RELEASE = process.env.PV_IQA_RS_RELEASE !== "0"
const RUST_BINARY_OVERRIDE = (process.env.PV_IQA_RS_BINARY ?? "").trim()
const RUST_BINARY_NAME = process.platform === "win32" ? "pv-iqa-rs.exe" : "pv-iqa-rs"
const RUST_BINARY_PROFILE = RUST_USE_RELEASE ? "release" : "debug"
const RUST_DEVICE_PREFERENCE = (process.env.PV_IQA_RS_DEVICE ?? "auto").trim().toLowerCase()
const RUST_PREPROCESS_MODE = (process.env.PV_IQA_RS_PREPROCESS_MODE ?? "python-pillow").trim()
const RUST_RESIZE_MODE = (process.env.PV_IQA_RS_RESIZE_MODE ?? "fast-convolution-bilinear").trim()
const RUST_BATCH_CONCURRENCY = Math.max(1, Math.min(4, Number(process.env.PV_IQA_RS_BATCH_CONCURRENCY ?? 2) || 2))
const RUST_BATCH_CHUNK_SIZE = Math.max(1, Math.min(32, Number(process.env.PV_IQA_RS_BATCH_CHUNK_SIZE ?? 16) || 16))
const RUST_HEALTH_TIMEOUT_MS = Number(process.env.PV_IQA_RS_HEALTH_TIMEOUT_MS ?? 1500)
const RUST_REQUEST_TIMEOUT_MS = Number(process.env.PV_IQA_RS_REQUEST_TIMEOUT_MS ?? 300000)
const RUST_START_TIMEOUT_MS = Number(process.env.PV_IQA_RS_START_TIMEOUT_MS ?? 300000)
const PYTHON_HEALTH_CACHE_MS = Number(process.env.PV_IQA_PY_HEALTH_CACHE_MS ?? 30000)
const PYTHON_HEALTH_SCRIPT = `
import json
import torch
from pv_iqa.config import load_config
from pv_iqa.utils.common import resolve_device

config = load_config("configs/default.yaml")
device = resolve_device(config)
payload = {
    "device": str(device),
    "runtime_device": config.runtime.device,
    "cuda_available": torch.cuda.is_available(),
    "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
}
print(json.dumps(payload))
`.trim()

let rustServiceStartup: Promise<void> | null = null
let rustStartupError: string | null = null
let pythonHealthCache:
  | {
      expiresAt: number
      result: {
        available: boolean
        payload?: PythonHealthPayload
        error?: string
      }
    }
  | null = null
const activeJobs = new Map<string, ActiveJobContext>()

class JobStoppedError extends Error {
  constructor(message: string) {
    super(message)
    this.name = "JobStoppedError"
  }
}

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
ensureColumn("jobs", "backend", "backend TEXT NOT NULL DEFAULT 'python'")

db.exec(`
  UPDATE jobs
  SET processed_count = COALESCE(processed_count, 0),
      progress = COALESCE(progress, 0),
      stage = CASE WHEN stage IS NULL OR stage = '' THEN '等待中' ELSE stage END,
      updated_at = CASE WHEN updated_at IS NULL OR updated_at = '' THEN created_at ELSE updated_at END,
      backend = CASE WHEN backend IS NULL OR backend = '' THEN 'python' ELSE backend END;
`)

const insertJobQuery = db.query(
  "INSERT INTO jobs (id, kind, status, backend, run_name, input_count, processed_count, progress, stage, created_at, updated_at, completed_at, error) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)
const updateJobStateQuery = db.query(
  "UPDATE jobs SET status = ?, processed_count = ?, progress = ?, stage = ?, updated_at = ?, completed_at = ?, error = ? WHERE id = ?"
)
const insertResultQuery = db.query(
  "INSERT INTO results (job_id, image_path, relative_path, public_url, quality_score) VALUES (?, ?, ?, ?, ?)"
)
const deleteResultsQuery = db.query("DELETE FROM results WHERE job_id = ?")
const deleteJobQuery = db.query("DELETE FROM jobs WHERE id = ?")
const countResultsQuery = db.query("SELECT COUNT(*) AS count FROM results WHERE job_id = ?")
const listResultPathsQuery = db.query("SELECT relative_path FROM results WHERE job_id = ? ORDER BY relative_path ASC")

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

function createActiveJobContext(
  jobId: string,
  kind: JobKind,
  backend: InferenceBackend,
  runName: string
): ActiveJobContext {
  return {
    jobId,
    kind,
    backend,
    runName,
    abortController: new AbortController(),
    currentPythonProcess: null,
    stopRequested: false,
    stopReason: null,
    done: null,
  }
}

function assertJobActive(jobContext?: ActiveJobContext): void {
  if (!jobContext) {
    return
  }
  if (jobContext.stopRequested || jobContext.abortController.signal.aborted) {
    throw new JobStoppedError(jobContext.stopReason || "任务已停止，可继续处理。")
  }
}

function requestJobStop(jobId: string, reason: string): ActiveJobContext | null {
  const jobContext = activeJobs.get(jobId)
  if (!jobContext) {
    return null
  }

  jobContext.stopRequested = true
  jobContext.stopReason = reason
  if (!jobContext.abortController.signal.aborted) {
    jobContext.abortController.abort(reason)
  }
  try {
    jobContext.currentPythonProcess?.kill()
  } catch {
    // Ignore transient kill failures; the stopped flag still prevents new work.
  }
  return jobContext
}

function startManagedJob(
  jobId: string,
  kind: JobKind,
  backend: InferenceBackend,
  runName: string,
  execute: (jobContext: ActiveJobContext) => Promise<void>
): void {
  if (activeJobs.has(jobId)) {
    throw new Error(`Job ${jobId} is already running.`)
  }
  const jobContext = createActiveJobContext(jobId, kind, backend, runName)
  activeJobs.set(jobId, jobContext)
  const done = execute(jobContext).finally(() => {
    if (activeJobs.get(jobId) === jobContext) {
      activeJobs.delete(jobId)
    }
  })
  jobContext.done = done
  void done
}

function resultCountForJob(jobId: string): number {
  const row = countResultsQuery.get(jobId) as { count: number } | null
  return Number(row?.count ?? 0)
}

function loadResultPaths(jobId: string): Set<string> {
  const rows = listResultPathsQuery.all(jobId) as Array<{ relative_path: string }>
  return new Set(rows.map((row) => row.relative_path))
}

async function collectSavedUploads(jobId: string, currentPath: string, prefix = ""): Promise<SavedUpload[]> {
  const entries = await readdir(currentPath, { withFileTypes: true }).catch(() => [])
  const uploads: SavedUpload[] = []

  for (const entry of entries) {
    const relativePath = prefix ? `${prefix}/${entry.name}` : entry.name
    const savedPath = resolve(currentPath, entry.name)
    if (entry.isDirectory()) {
      uploads.push(...(await collectSavedUploads(jobId, savedPath, relativePath)))
      continue
    }
    if (!entry.isFile()) {
      continue
    }
    const cleanedPath = sanitizeRelativePath(relativePath)
    uploads.push({
      savedPath,
      relativePath: cleanedPath,
      publicUrl: publicUploadUrl(jobId, cleanedPath),
    })
  }

  return uploads
}

async function loadSavedUploads(jobId: string): Promise<SavedUpload[]> {
  const uploads = await collectSavedUploads(jobId, resolve(UPLOAD_ROOT, jobId))
  return uploads.sort((left, right) => left.relativePath.localeCompare(right.relativePath, "en"))
}

function interruptedStage(kind: JobKind, processedCount: number, total: number, reason: string): string {
  if (kind === "folder") {
    return `${reason} · ${processedCount}/${total}`
  }
  return reason
}

function finalizeInterruptedJob(jobId: string, kind: JobKind, total: number, reason: string): void {
  const processedCount = Math.min(total, resultCountForJob(jobId))
  const completed = total > 0 && processedCount >= total
  persistJobState(jobId, {
    status: completed ? "completed" : "interrupted",
    processedCount,
    progress: total ? (processedCount / total) * 100 : 0,
    stage: completed
      ? kind === "image"
        ? "评分完成"
        : "批量评分完成"
      : interruptedStage(kind, processedCount, total, reason),
    completedAt: completed ? nowIso() : null,
    error: completed ? null : reason,
  })
}

function finalizeFailedJob(jobId: string, kind: JobKind, total: number, reason: string): void {
  const processedCount = Math.min(total, resultCountForJob(jobId))
  persistJobState(jobId, {
    status: "failed",
    processedCount,
    progress: total ? (processedCount / total) * 100 : 0,
    stage: kind === "image" ? "评分失败" : "批量评分失败",
    completedAt: nowIso(),
    error: reason,
  })
}

function reconcileRunningJobsOnStartup(): void {
  const jobs = db
    .query(
      `SELECT id, kind, backend, run_name, input_count, processed_count, progress, stage, created_at, updated_at, completed_at, error
       FROM jobs
       WHERE status = 'running'
       ORDER BY created_at ASC`
    )
    .all() as Array<Omit<JobSummary, "status" | "result_count" | "average_score" | "best_score" | "worst_score">>

  for (const job of jobs) {
    const processedCount = Math.min(job.input_count, resultCountForJob(job.id))
    const completed = job.input_count > 0 && processedCount >= job.input_count
    persistJobState(job.id, {
      status: completed ? "completed" : "interrupted",
      processedCount,
      progress: completed ? 100 : job.input_count ? (processedCount / job.input_count) * 100 : 0,
      stage: completed
        ? job.kind === "image"
          ? "评分完成（服务恢复）"
          : "批量评分完成（服务恢复）"
        : interruptedStage(job.kind, processedCount, job.input_count, "服务重启后中断，可继续处理"),
      completedAt: completed ? job.completed_at ?? nowIso() : null,
      error: completed ? null : "任务在服务停止时未正常结束。",
    })
  }
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

async function runPvIqaCommand(
  args: string[],
  jobContext?: ActiveJobContext
): Promise<{ stdout: string; stderr: string }> {
  assertJobActive(jobContext)
  const command = Bun.spawn(["uv", "run", "pv-iqa", ...args], {
    cwd: REPO_ROOT,
    stdout: "pipe",
    stderr: "pipe",
    env: {
      ...process.env,
      PYTHONIOENCODING: "utf-8",
    },
  })
  if (jobContext) {
    jobContext.currentPythonProcess = command
  }

  const stdoutPromise = new Response(command.stdout).text()
  const stderrPromise = new Response(command.stderr).text()
  try {
    const exitCode = await command.exited
    const stdout = (await stdoutPromise).trim()
    const stderr = (await stderrPromise).trim()

    if (jobContext?.stopRequested) {
      throw new JobStoppedError(jobContext.stopReason || "任务已停止，可继续处理。")
    }

    if (exitCode !== 0) {
      throw new Error(stderr || stdout || "pv-iqa command failed")
    }

    return { stdout, stderr }
  } finally {
    if (jobContext?.currentPythonProcess === command) {
      jobContext.currentPythonProcess = null
    }
  }
}

async function runPvIqaJson(args: string[], jobContext?: ActiveJobContext): Promise<unknown> {
  const { stdout } = await runPvIqaCommand(args, jobContext)

  try {
    return JSON.parse(stdout)
  } catch {
    throw new Error(`Failed to parse pv-iqa JSON output: ${stdout}`)
  }
}

function rustBinaryTargetPath(): string {
  return resolve(RUST_PROJECT_ROOT, "target", RUST_BINARY_PROFILE, RUST_BINARY_NAME)
}

function rustBinaryTimestamp(): string {
  const iso = new Date().toISOString()
  return `${iso.slice(0, 10).replace(/-/g, "")}-${iso.slice(11, 19).replace(/:/g, "")}`
}

type RustBinaryVariant = "cpu" | "cuda"

function rustBinaryPrefix(variant: RustBinaryVariant): string {
  return `pv-iqa-rs-${RUST_BINARY_PROFILE}-${variant}-`
}

function rustBinaryFeatureArgs(variant: RustBinaryVariant): string[] {
  return variant === "cuda" ? ["--features", RUST_CUDA_FEATURES] : []
}

function preferredRustBinaryVariant(): RustBinaryVariant {
  return RUST_DEVICE_PREFERENCE === "cpu" ? "cpu" : "cuda"
}

async function findLatestRustBinary(variant: RustBinaryVariant): Promise<string | null> {
  if (RUST_BINARY_OVERRIDE) {
    const overridePath = resolve(REPO_ROOT, RUST_BINARY_OVERRIDE)
    if (!(await Bun.file(overridePath).exists())) {
      throw new Error(`Configured Rust binary not found: ${overridePath}`)
    }
    return overridePath
  }

  await mkdir(BIN_ROOT, { recursive: true })
  const entries = await readdir(BIN_ROOT, { withFileTypes: true }).catch(() => [])
  const candidates = await Promise.all(
    entries
      .filter((entry) => entry.isFile() && entry.name.startsWith(rustBinaryPrefix(variant)))
      .map(async (entry) => {
        const filePath = resolve(BIN_ROOT, entry.name)
        const file = Bun.file(filePath)
        if (!(await file.exists())) {
          return null
        }
        return {
          path: filePath,
          updatedAt: file.lastModified,
        }
      })
  )

  const latest = candidates
    .filter((item): item is { path: string; updatedAt: number } => item !== null)
    .sort((left, right) => right.updatedAt - left.updatedAt)[0]
  return latest?.path ?? null
}

async function publishRustBinary(variant: RustBinaryVariant, sourcePath: string): Promise<string> {
  await mkdir(BIN_ROOT, { recursive: true })
  const extension = process.platform === "win32" ? ".exe" : ""
  const targetPath = resolve(
    BIN_ROOT,
    `${rustBinaryPrefix(variant)}${rustBinaryTimestamp()}${extension}`
  )
  await copyFile(sourcePath, targetPath)
  await chmod(targetPath, 0o755).catch(() => undefined)
  return targetPath
}

async function buildRustBinary(variant: RustBinaryVariant): Promise<string> {
  if (!(await Bun.file(resolve(RUST_PROJECT_ROOT, "Cargo.toml")).exists())) {
    throw new Error(`Rust backend project not found at ${RUST_PROJECT_ROOT}.`)
  }

  console.log(
    `[pv-iqa] building Rust ${variant.toUpperCase()} binary (${RUST_BINARY_PROFILE}${variant === "cuda" ? `, features=${RUST_CUDA_FEATURES}` : ""})`
  )
  const buildCommand = Bun.spawn(
    [
      "cargo",
      "build",
      ...(RUST_USE_RELEASE ? ["--release"] : []),
      ...rustBinaryFeatureArgs(variant),
    ],
    {
      cwd: RUST_PROJECT_ROOT,
      stdout: "inherit",
      stderr: "inherit",
      env: {
        ...process.env,
        PV_IQA_REPO_ROOT: REPO_ROOT,
        PV_IQA_RS_HOST: RUST_SERVICE_HOST,
        PV_IQA_RS_PORT: String(RUST_SERVICE_PORT),
      },
    }
  )

  const exitCode = await buildCommand.exited
  if (exitCode !== 0) {
    throw new Error(`Rust binary build failed with exit code ${exitCode}.`)
  }

  const targetPath = rustBinaryTargetPath()
  if (!(await Bun.file(targetPath).exists())) {
    throw new Error(`Built Rust binary not found at ${targetPath}.`)
  }
  return publishRustBinary(variant, targetPath)
}

async function ensureRustBinaries(): Promise<Record<RustBinaryVariant, string>> {
  const variants: RustBinaryVariant[] = ["cpu", "cuda"]
  const binaries = {} as Record<RustBinaryVariant, string>

  for (const variant of variants) {
    const existingBinary = await findLatestRustBinary(variant)
    binaries[variant] = existingBinary ?? (await buildRustBinary(variant))
  }
  return binaries
}

async function ensureRustBinary(): Promise<string> {
  if (RUST_BINARY_OVERRIDE) {
    const overridePath = await findLatestRustBinary("cpu")
    if (!overridePath) {
      throw new Error("Configured Rust binary override could not be resolved.")
    }
    return overridePath
  }

  const binaries = await ensureRustBinaries()
  return binaries[preferredRustBinaryVariant()]
}

async function fetchJson<T>(
  url: string,
  init: RequestInit = {},
  timeoutMs = RUST_HEALTH_TIMEOUT_MS
): Promise<{ response: Response; payload: T & { error?: string } }> {
  const timeoutSignal = AbortSignal.timeout(timeoutMs)
  const signal = init.signal ? AbortSignal.any([init.signal, timeoutSignal]) : timeoutSignal
  const response = await fetch(url, {
    ...init,
    signal,
  })
  const payload = (await response.json()) as T & { error?: string }
  return { response, payload }
}

async function probeRustBackend(): Promise<{
  available: boolean
  payload?: RustHealthPayload
  error?: string
}> {
  try {
    const { response, payload } = await fetchJson<RustHealthPayload>(`${RUST_SERVICE_URL}/health`)
    if (!response.ok || payload.status !== "ok") {
      return {
        available: false,
        error: payload.error || `Rust backend health check failed (${response.status})`,
      }
    }
    rustStartupError = null
    return { available: true, payload }
  } catch (error) {
    return { available: false, error: String(error) }
  }
}

async function probePythonBackend(): Promise<{
  available: boolean
  payload?: PythonHealthPayload
  error?: string
}> {
  if (pythonHealthCache && pythonHealthCache.expiresAt > Date.now()) {
    return pythonHealthCache.result
  }

  const command = Bun.spawn(["uv", "run", "python", "-c", PYTHON_HEALTH_SCRIPT], {
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

  let result:
    | {
        available: boolean
        payload?: PythonHealthPayload
        error?: string
      }
    | undefined

  if (exitCode === 0) {
    try {
      result = {
        available: true,
        payload: JSON.parse(stdout) as PythonHealthPayload,
      }
    } catch {
      result = {
        available: false,
        error: `Failed to parse Python backend probe output: ${stdout}`,
      }
    }
  } else {
    result = {
      available: false,
      error: stderr || stdout || "Python backend probe failed",
    }
  }

  pythonHealthCache = {
    expiresAt: Date.now() + PYTHON_HEALTH_CACHE_MS,
    result,
  }
  return result
}

async function ensureRustModelArtifacts(runName: string): Promise<void> {
  const iqaRoot = resolve(REPO_ROOT, "checkpoints", runName, "iqa")
  const onnxPath = resolve(iqaRoot, "best.onnx")
  const metadataPath = resolve(iqaRoot, "best.onnx.json")
  const externalDataPath = resolve(iqaRoot, "best.onnx.data")
  const metadataFile = Bun.file(metadataPath)
  const metadataExists = await metadataFile.exists()
  let profileMismatch = true

  if (metadataExists) {
    try {
      const metadata = JSON.parse(await metadataFile.text()) as {
        export_profile?: string
      }
      profileMismatch = metadata.export_profile !== RUST_EXPORT_PROFILE
    } catch {
      profileMismatch = true
    }
  }

  const needsExport =
    !(await Bun.file(onnxPath).exists()) ||
    !metadataExists ||
    (await Bun.file(externalDataPath).exists()) ||
    profileMismatch

  if (!needsExport) {
    return
  }

  await runPvIqaCommand(["export-rust-model", "--run-name", runName])
}

async function startRustBackend(): Promise<void> {
  if (!RUST_AUTOSTART) {
    throw new Error(
      `Rust backend is unavailable at ${RUST_SERVICE_URL}. Start the service from ${RUST_PROJECT_ROOT} and retry.`
    )
  }

  if (rustServiceStartup) {
    return rustServiceStartup
  }

  const startup = (async () => {
    rustStartupError = null
    const binaryPath = await ensureRustBinary()
    Bun.spawn(
      [binaryPath],
      {
        cwd: REPO_ROOT,
        stdout: "inherit",
        stderr: "inherit",
        env: {
          ...process.env,
          PV_IQA_REPO_ROOT: REPO_ROOT,
          PV_IQA_RS_HOST: RUST_SERVICE_HOST,
          PV_IQA_RS_PORT: String(RUST_SERVICE_PORT),
          PV_IQA_RS_DEVICE: process.env.PV_IQA_RS_DEVICE ?? "auto",
          PV_IQA_RS_PREPROCESS_MODE: RUST_PREPROCESS_MODE,
          PV_IQA_RS_RESIZE_MODE: RUST_RESIZE_MODE,
        },
      }
    )

    const deadline = Date.now() + RUST_START_TIMEOUT_MS
    while (Date.now() < deadline) {
      await Bun.sleep(500)
      const health = await probeRustBackend()
      if (health.available) {
        return
      }
    }

    throw new Error(
      `Rust backend did not become ready within ${Math.round(RUST_START_TIMEOUT_MS / 1000)}s.`
    )
  })()
  rustServiceStartup = startup
  startup
    .catch((error) => {
      rustStartupError = String(error)
      console.error(`[pv-iqa] Rust backend startup failed: ${rustStartupError}`)
    })
    .finally(() => {
      if (rustServiceStartup === startup) {
        rustServiceStartup = null
      }
    })

  return startup
}

function warmRustBackendInBackground(): void {
  if (!RUST_AUTOSTART || rustServiceStartup || rustStartupError) {
    return
  }
  void startRustBackend().catch(() => undefined)
}

async function ensureRustBackendReady(runName: string): Promise<RustHealthPayload> {
  await ensureRustModelArtifacts(runName)

  let health = await probeRustBackend()
  if (!health.available) {
    await startRustBackend()
    health = await probeRustBackend()
  }

  if (!health.available || !health.payload) {
    throw new Error(health.error || "Rust backend is unavailable.")
  }
  return health.payload
}

async function callRustBackend<T>(path: string, payload: unknown, signal?: AbortSignal): Promise<T> {
  const { response, payload: body } = await fetchJson<T>(`${RUST_SERVICE_URL}${path}`, {
    method: "POST",
    headers: {
      "content-type": "application/json",
    },
    body: JSON.stringify(payload),
    signal,
  }, RUST_REQUEST_TIMEOUT_MS)

  if (!response.ok) {
    throw new Error(body.error || `Rust backend request failed (${response.status}).`)
  }

  return body
}

function clampProgress(value: number): number {
  return Math.max(0, Math.min(100, value))
}

function createJob(
  jobId: string,
  kind: JobKind,
  backend: InferenceBackend,
  runName: string,
  inputCount: number,
  state: JobState
): void {
  const timestamp = nowIso()
  insertJobQuery.run(
    jobId,
    kind,
    state.status,
    backend,
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

reconcileRunningJobsOnStartup()

function loadJobSummary(jobId: string): JobSummary | null {
  return db
    .query(
      `SELECT
         jobs.id,
         jobs.kind,
         jobs.status,
         jobs.backend,
         jobs.run_name,
         jobs.input_count,
         jobs.processed_count,
         jobs.progress,
         jobs.stage,
         jobs.created_at,
         jobs.updated_at,
         jobs.completed_at,
         jobs.error,
         (SELECT COUNT(*) FROM results WHERE job_id = jobs.id) AS result_count,
         (SELECT AVG(quality_score) FROM results WHERE job_id = jobs.id) AS average_score,
         (SELECT MAX(quality_score) FROM results WHERE job_id = jobs.id) AS best_score,
         (SELECT MIN(quality_score) FROM results WHERE job_id = jobs.id) AS worst_score
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
         jobs.backend,
         jobs.run_name,
         jobs.input_count,
         jobs.processed_count,
         jobs.progress,
         jobs.stage,
         jobs.created_at,
         jobs.updated_at,
         jobs.completed_at,
         jobs.error,
         (SELECT COUNT(*) FROM results WHERE job_id = jobs.id) AS result_count,
         (SELECT AVG(quality_score) FROM results WHERE job_id = jobs.id) AS average_score,
         (SELECT MAX(quality_score) FROM results WHERE job_id = jobs.id) AS best_score,
         (SELECT MIN(quality_score) FROM results WHERE job_id = jobs.id) AS worst_score
       FROM jobs
       ORDER BY jobs.created_at DESC
       LIMIT ?`
    )
    .all(limit) as JobSummary[]
}

async function detectImageWithPython(
  savedPath: string,
  runName: string,
  jobContext?: ActiveJobContext
): Promise<ImageDetectionPayload["result"]> {
  const output = (await runPvIqaJson([
    "detect-image",
    savedPath,
    "--run-name",
    runName,
    "--json-output",
  ], jobContext)) as ImageDetectionPayload
  return output.result
}

async function detectImageWithRust(
  savedPath: string,
  runName: string,
  jobContext?: ActiveJobContext
): Promise<RustImageDetectionPayload["result"]> {
  assertJobActive(jobContext)
  await ensureRustBackendReady(runName)
  return detectImageWithRustReady(savedPath, runName, jobContext?.abortController.signal)
}

async function detectImageWithRustReady(
  savedPath: string,
  runName: string,
  signal?: AbortSignal
): Promise<RustImageDetectionPayload["result"]> {
  const output = await callRustBackend<RustImageDetectionPayload>("/score/image", {
    run_name: runName,
    image_path: savedPath,
  }, signal)
  return output.result
}

async function detectBatchWithRust(
  uploads: SavedUpload[],
  runName: string,
  onProgress?: (update: {
    upload: SavedUpload
    result: RustImageDetectionPayload["result"]
    completedCount: number
    total: number
  }) => void,
  signal?: AbortSignal
): Promise<RustBatchDetectionPayload["results"]> {
  await ensureRustBackendReady(runName)
  if (!uploads.length) {
    return []
  }

  const uploadChunks = Array.from({ length: Math.ceil(uploads.length / RUST_BATCH_CHUNK_SIZE) }, (_, chunkIndex) =>
    uploads
      .slice(chunkIndex * RUST_BATCH_CHUNK_SIZE, (chunkIndex + 1) * RUST_BATCH_CHUNK_SIZE)
      .map((upload, indexInChunk) => ({
        upload,
        index: chunkIndex * RUST_BATCH_CHUNK_SIZE + indexInChunk,
      }))
  )
  const results = new Array<RustImageDetectionPayload["result"] | undefined>(uploads.length)
  const workerCount = Math.min(RUST_BATCH_CONCURRENCY, uploadChunks.length)
  let nextChunkIndex = 0
  let completedCount = 0

  async function detectBatchChunkWithRustReady(
    chunk: SavedUpload[],
    currentRunName: string
  ): Promise<RustBatchDetectionPayload["results"]> {
    const output = await callRustBackend<RustBatchDetectionPayload>("/score/batch", {
      run_name: currentRunName,
      image_paths: chunk.map((upload) => upload.savedPath),
    }, signal)
    return output.results
  }

  async function worker(): Promise<void> {
    while (nextChunkIndex < uploadChunks.length) {
      const currentChunkIndex = nextChunkIndex
      nextChunkIndex += 1
      const chunkEntries = uploadChunks[currentChunkIndex]
      const chunkUploads = chunkEntries.map((entry) => entry.upload)
      const chunkResults = await detectBatchChunkWithRustReady(chunkUploads, runName)
      if (chunkResults.length !== chunkEntries.length) {
        throw new Error(
          `Rust backend returned ${chunkResults.length} results for chunk size ${chunkEntries.length}.`
        )
      }

      const entryByPath = new Map(chunkEntries.map((entry) => [entry.upload.savedPath, entry] as const))
      for (const [resultIndex, result] of chunkResults.entries()) {
        const entry = entryByPath.get(result.image_path) ?? chunkEntries[resultIndex]
        if (!entry) {
          throw new Error(`Rust backend returned an unknown image path: ${result.image_path}`)
        }
        results[entry.index] = result
        completedCount += 1
        onProgress?.({
          upload: entry.upload,
          result,
          completedCount,
          total: uploads.length,
        })
      }
    }
  }

  await Promise.all(Array.from({ length: workerCount }, () => worker()))
  return results.map((result, index) => {
    if (!result) {
      throw new Error(`Rust backend returned no result for upload ${uploads[index]?.savedPath ?? index}.`)
    }
    return result
  })
}

async function detectImage(
  savedPath: string,
  runName: string,
  backend: InferenceBackend,
  jobContext?: ActiveJobContext
): Promise<ImageDetectionPayload["result"]> {
  if (backend === "rust") {
    return detectImageWithRust(savedPath, runName, jobContext)
  }
  return detectImageWithPython(savedPath, runName, jobContext)
}

type FolderProcessOptions = {
  startingProcessedCount?: number
  skipRelativePaths?: Set<string>
  resumeLabel?: string
}

async function processImageJob(
  jobId: string,
  runName: string,
  upload: SavedUpload,
  backend: InferenceBackend,
  jobContext: ActiveJobContext
): Promise<void> {
  const state: JobState = {
    status: "running",
    processedCount: 0,
    progress: 0,
    stage: backend === "rust" ? "Rust 后端评分中" : "模型评分中",
    completedAt: null,
    error: null,
  }
  persistJobState(jobId, state)

  try {
    const result = await detectImage(upload.savedPath, runName, backend, jobContext)
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
    if (jobContext.stopRequested || error instanceof JobStoppedError) {
      finalizeInterruptedJob(jobId, "image", 1, jobContext.stopReason || "任务已停止，可继续处理")
      return
    }
    finalizeFailedJob(jobId, "image", 1, String(error))
  }
}

async function processFolderJob(
  jobId: string,
  runName: string,
  uploads: SavedUpload[],
  backend: InferenceBackend,
  jobContext: ActiveJobContext,
  options: FolderProcessOptions = {}
): Promise<void> {
  const total = uploads.length
  const completedPaths = options.skipRelativePaths ?? new Set<string>()
  const pendingUploads = uploads.filter((upload) => !completedPaths.has(upload.relativePath))
  const startingProcessedCount = Math.min(options.startingProcessedCount ?? completedPaths.size, total)
  const state: JobState = {
    status: "running",
    processedCount: startingProcessedCount,
    progress: total ? (startingProcessedCount / total) * 100 : 100,
    stage: total
      ? options.resumeLabel ?? (startingProcessedCount ? `继续批量评分 · ${startingProcessedCount}/${total}` : "等待批量评分")
      : "没有可评分图片",
    completedAt: null,
    error: null,
  }
  persistJobState(jobId, state)

  try {
    assertJobActive(jobContext)
    if (!pendingUploads.length) {
      state.status = "completed"
      state.processedCount = total
      state.progress = 100
      state.stage = "批量评分完成"
      state.completedAt = nowIso()
      persistJobState(jobId, state)
      return
    }

    if (backend === "rust") {
      const workerCount = Math.min(Math.ceil(pendingUploads.length / RUST_BATCH_CHUNK_SIZE), RUST_BATCH_CONCURRENCY)
      state.stage = pendingUploads.length
        ? `Rust 分块批量评分中 · ${startingProcessedCount}/${total}（${RUST_BATCH_CHUNK_SIZE}/批，${workerCount} 路）`
        : "没有可评分图片"
      state.progress = total ? (startingProcessedCount / total) * 100 : 100
      persistJobState(jobId, state)

      const results = await detectBatchWithRust(
        pendingUploads,
        runName,
        ({ upload, result, completedCount }) => {
        insertResultQuery.run(
          jobId,
          result.image_path,
          upload.relativePath,
          upload.publicUrl,
          result.quality_score
        )

          state.processedCount = startingProcessedCount + completedCount
          state.progress = total ? (state.processedCount / total) * 100 : 100
        state.stage =
            state.processedCount < total
              ? `Rust 分块批量评分中 · ${state.processedCount}/${total}（${RUST_BATCH_CHUNK_SIZE}/批，${workerCount} 路）`
              : `已完成 ${state.processedCount}/${total}`
        persistJobState(jobId, state)
        },
        jobContext.abortController.signal
      )
      if (results.length !== pendingUploads.length) {
        throw new Error(`Rust backend returned ${results.length} results for ${pendingUploads.length} uploads.`)
      }

      state.status = "completed"
      state.processedCount = total
      state.progress = 100
      state.stage = "批量评分完成"
      state.completedAt = nowIso()
      persistJobState(jobId, state)
      return
    }

    for (const [index, upload] of pendingUploads.entries()) {
      assertJobActive(jobContext)
      state.stage = `批量评分中 ${startingProcessedCount + index + 1}/${total}`
      state.progress = total ? ((startingProcessedCount + index) / total) * 100 : 100
      persistJobState(jobId, state)

      const result = await detectImage(upload.savedPath, runName, backend, jobContext)
      insertResultQuery.run(
        jobId,
        result.image_path,
        upload.relativePath,
        upload.publicUrl,
        result.quality_score
      )

      state.processedCount = startingProcessedCount + index + 1
      state.progress = total ? (state.processedCount / total) * 100 : 100
      state.stage = `已完成 ${state.processedCount}/${total}`
      persistJobState(jobId, state)
    }

    state.status = "completed"
    state.processedCount = total
    state.progress = 100
    state.stage = "批量评分完成"
    state.completedAt = nowIso()
    persistJobState(jobId, state)
  } catch (error) {
    if (jobContext.stopRequested || error instanceof JobStoppedError) {
      finalizeInterruptedJob(jobId, "folder", total, jobContext.stopReason || "任务已停止，可继续处理")
      return
    }
    finalizeFailedJob(jobId, "folder", total, String(error))
  }
}

async function parseRunName(formData: FormDataLike): Promise<string> {
  const rawRunName = String(formData.get("runName") || "").trim()
  return rawRunName || (await resolveDefaultRunName())
}

function parseBackend(formData: FormDataLike): InferenceBackend {
  const rawBackend = String(formData.get("backend") || "python")
    .trim()
    .toLowerCase()

  if (rawBackend === "python" || rawBackend === "rust") {
    return rawBackend
  }
  throw new Error("Invalid inference backend.")
}

async function requireSavedJobUploads(jobId: string): Promise<SavedUpload[]> {
  const uploads = await loadSavedUploads(jobId)
  if (!uploads.length) {
    throw new Error("未找到该任务对应的原始上传文件。")
  }
  return uploads
}

function startExistingJob(job: JobSummary, uploads: SavedUpload[], options?: FolderProcessOptions): void {
  if (job.kind === "image") {
    const upload = uploads[0]
    if (!upload) {
      throw new Error("未找到用于重新评分的图片文件。")
    }
    startManagedJob(job.id, job.kind, job.backend, job.run_name, (jobContext) =>
      processImageJob(job.id, job.run_name, upload, job.backend, jobContext)
    )
    return
  }

  startManagedJob(job.id, job.kind, job.backend, job.run_name, (jobContext) =>
    processFolderJob(job.id, job.run_name, uploads, job.backend, jobContext, options)
  )
}

async function handleStopJob(jobId: string): Promise<Response> {
  const job = loadJobSummary(jobId)
  if (!job) {
    return errorResponse("Job not found.", 404)
  }
  if (job.status !== "running") {
    return errorResponse("Only running jobs can be stopped.", 409)
  }

  const runtime = requestJobStop(jobId, "任务已停止，可继续处理")
  if (runtime) {
    await runtime.done?.catch(() => undefined)
  } else {
    finalizeInterruptedJob(jobId, job.kind, job.input_count, "任务已停止，可继续处理")
  }

  return jsonResponse({ job: loadJob(jobId) })
}

async function handleResumeJob(jobId: string): Promise<Response> {
  const job = loadJobSummary(jobId)
  if (!job) {
    return errorResponse("Job not found.", 404)
  }
  if (job.status !== "interrupted") {
    return errorResponse("Only interrupted jobs can be resumed.", 409)
  }
  if (activeJobs.has(jobId)) {
    return errorResponse("This job is already running.", 409)
  }

  try {
    const uploads = await requireSavedJobUploads(jobId)
    if (job.kind === "image") {
      if (resultCountForJob(jobId) >= 1) {
        finalizeInterruptedJob(jobId, "image", 1, "评分已完成")
        return jsonResponse({ job: loadJob(jobId) })
      }
      startExistingJob(job, uploads)
      return jsonResponse({ job: loadJob(jobId) }, { status: 202 })
    }

    const completedPaths = loadResultPaths(jobId)
    startExistingJob(job, uploads, {
      startingProcessedCount: completedPaths.size,
      skipRelativePaths: completedPaths,
      resumeLabel: `继续批量评分 · ${completedPaths.size}/${uploads.length}`,
    })
    return jsonResponse({ job: loadJob(jobId) }, { status: 202 })
  } catch (error) {
    return errorResponse(String(error), 409)
  }
}

async function handleRerunJob(jobId: string): Promise<Response> {
  const job = loadJobSummary(jobId)
  if (!job) {
    return errorResponse("Job not found.", 404)
  }
  if (job.status === "running" || activeJobs.has(jobId)) {
    return errorResponse("Running jobs must be stopped before rerun.", 409)
  }

  try {
    const uploads = await requireSavedJobUploads(jobId)
    deleteResultsQuery.run(jobId)
    startExistingJob(job, uploads)
    return jsonResponse({ job: loadJob(jobId) }, { status: 202 })
  } catch (error) {
    return errorResponse(String(error), 409)
  }
}

async function handleImageScore(request: Request): Promise<Response> {
  const formData = await request.formData()
  const file = formData.get("file") as MultipartEntry | null
  if (!file || !isUploadedFile(file)) {
    return errorResponse("Missing image file.")
  }

  try {
    const runName = await parseRunName(formData)
    const backend = parseBackend(formData)
    const jobId = crypto.randomUUID()
    const upload = await saveUploadedFile(jobId, file.name, file)

    createJob(jobId, "image", backend, runName, 1, {
      status: "running",
      processedCount: 0,
      progress: 0,
      stage: "上传完成，等待评分",
      completedAt: null,
      error: null,
    })

    startManagedJob(jobId, "image", backend, runName, (jobContext) =>
      processImageJob(jobId, runName, upload, backend, jobContext)
    )
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
    const backend = parseBackend(formData)
    const jobId = crypto.randomUUID()
    const uploads: SavedUpload[] = []

    for (const [index, file] of files.entries()) {
      const relativePath = manifest[index]?.relativePath || file.name
      uploads.push(await saveUploadedFile(jobId, relativePath, file))
    }

    createJob(jobId, "folder", backend, runName, uploads.length, {
      status: "running",
      processedCount: 0,
      progress: 0,
      stage: "上传完成，等待批量评分",
      completedAt: null,
      error: null,
    })

    startManagedJob(jobId, "folder", backend, runName, (jobContext) =>
      processFolderJob(jobId, runName, uploads, backend, jobContext)
    )
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

  const runtime = activeJobs.get(jobId)
  if (job.status === "running" || runtime) {
    requestJobStop(jobId, "任务已强制停止，准备删除")
    await runtime?.done?.catch(() => undefined)
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
      const [defaultRunResult, pythonStatus, rustStatus] = await Promise.allSettled([
        resolveDefaultRunName(),
        probePythonBackend(),
        probeRustBackend(),
      ])
      const defaultRunName = defaultRunResult.status === "fulfilled" ? defaultRunResult.value : undefined
      const defaultRunError =
        defaultRunResult.status === "rejected" ? String(defaultRunResult.reason) : undefined
      const pythonBackend =
        pythonStatus.status === "fulfilled"
          ? pythonStatus.value
          : { available: false, error: String(pythonStatus.reason) }
      const rustBackend =
        rustStatus.status === "fulfilled"
          ? rustStatus.value
          : { available: false, error: String(rustStatus.reason) }
      if (!rustBackend.available) {
        warmRustBackendInBackground()
      }
      const rustState = rustBackend.available
        ? "ready"
        : rustServiceStartup
          ? "starting"
          : RUST_AUTOSTART && !rustStartupError
            ? "idle"
            : "error"

      return jsonResponse({
        status: defaultRunError || !pythonBackend.available ? "degraded" : "ok",
        defaultRunName,
        error: defaultRunError,
        port: PORT,
        backends: {
          python: {
            available: pythonBackend.available,
            label: "Python",
            detail: pythonBackend.available
              ? pythonBackend.payload?.gpu_name ?? `runtime=${pythonBackend.payload?.runtime_device ?? "auto"}`
              : "uv run pv-iqa",
            device: pythonBackend.payload?.device,
            error: pythonBackend.available ? undefined : pythonBackend.error,
          },
          rust: {
            available: rustBackend.available,
            label: "Rust Candle",
            state: rustState,
            detail: rustBackend.available
              ? `${rustBackend.payload?.service ?? "pv-iqa-rs"} / ${rustBackend.payload?.device ?? "cpu"}`
              : rustState === "starting"
                ? `正在启动，目标地址 ${RUST_SERVICE_URL}`
                : rustState === "idle"
                  ? "未启动；选择 Rust 任务时会自动拉起"
                  : RUST_SERVICE_URL,
            device: rustBackend.payload?.device,
            error:
              rustBackend.available || rustState === "starting" || rustState === "idle"
                ? undefined
                : rustStartupError ?? rustBackend.error,
          },
        },
      })
    }

    if (url.pathname === "/api/jobs" && request.method === "GET") {
      return jsonResponse({ jobs: listJobs() })
    }

    if (url.pathname.startsWith("/api/jobs/")) {
      const [, , jobId, action] = url.pathname.split("/").filter(Boolean)
      if (!jobId) {
        return errorResponse("Job not found.", 404)
      }

      if (!action && request.method === "GET") {
        const job = loadJob(jobId)
        return job ? jsonResponse({ job }) : errorResponse("Job not found.", 404)
      }

      if (!action && request.method === "DELETE") {
        return handleDeleteJob(jobId)
      }

      if (request.method === "POST") {
        if (action === "stop") {
          return handleStopJob(jobId)
        }
        if (action === "resume") {
          return handleResumeJob(jobId)
        }
        if (action === "rerun") {
          return handleRerunJob(jobId)
        }
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
