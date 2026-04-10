import type { BackendHealth, InferenceBackend, JobStatus, ScoreResult } from "@/lib/types"

export function formatTime(value: string | null): string {
  if (!value) {
    return "—"
  }
  return new Date(value).toLocaleString("zh-CN")
}

export function compactFileSize(bytes: number): string {
  if (bytes < 1024) {
    return `${bytes} B`
  }
  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`
  }
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`
}

export function statusText(status: JobStatus): string {
  if (status === "completed") {
    return "已完成"
  }
  if (status === "interrupted") {
    return "已中断"
  }
  if (status === "failed") {
    return "失败"
  }
  return "处理中"
}

export function statusVariant(status: JobStatus): "secondary" | "destructive" | "outline" {
  if (status === "completed") {
    return "secondary"
  }
  if (status === "failed") {
    return "destructive"
  }
  return "outline"
}

export function qualityLabel(score: number): string {
  if (score >= 0.85) {
    return "优秀"
  }
  if (score >= 0.65) {
    return "良好"
  }
  if (score >= 0.45) {
    return "一般"
  }
  return "偏低"
}

export function backendText(backend: InferenceBackend): string {
  return backend === "rust" ? "Rust" : "Python"
}

export function deviceText(device?: string | null): string {
  if (!device) {
    return "未知"
  }
  const normalized = device.toLowerCase()
  if (normalized.startsWith("cuda")) {
    return "CUDA"
  }
  if (normalized === "cpu") {
    return "CPU"
  }
  if (normalized === "mps") {
    return "MPS"
  }
  return device.toUpperCase()
}

export function backendDeviceText(health?: BackendHealth | null): string {
  if (!health) {
    return "检测中"
  }
  if (health.state === "starting") {
    return "启动中"
  }
  if (health.state === "idle") {
    return "待启动"
  }
  if (!health.available) {
    return "异常"
  }
  return deviceText(health.device)
}

export function backendHintText(health?: BackendHealth | null): string {
  if (!health) {
    return "正在检测后端状态。"
  }
  if (health.available) {
    return `${health.label} · ${deviceText(health.device)}${health.detail ? ` · ${health.detail}` : ""}`
  }
  if (health.state === "starting" || health.state === "idle") {
    return health.detail ?? `${health.label} 正在准备中。`
  }
  return `${health.label} 不可用${health.error ? ` · ${health.error}` : health.detail ? ` · ${health.detail}` : ""}`
}

export function averageScore(results: ScoreResult[]): number | null {
  if (!results.length) {
    return null
  }
  return results.reduce((sum, item) => sum + item.quality_score, 0) / results.length
}

export function formatScore(value: number | null | undefined, digits = 4): string {
  if (value == null || Number.isNaN(value)) {
    return "—"
  }
  return value.toFixed(digits)
}

export function clampPercentage(value: number): number {
  return Math.max(0, Math.min(100, value))
}
