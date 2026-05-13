import type { InferenceBackend, JobStatus, ScoreResult } from "@/lib/types"

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

export function qualityLabel(score: number, allScores: number[]): string {
  if (allScores.length < 2) {
    return score >= 50 ? "好" : score >= 30 ? "中" : "差"
  }
  const mean = allScores.reduce((a, b) => a + b, 0) / allScores.length
  const std = Math.sqrt(
    allScores.reduce((sum, s) => sum + (s - mean) ** 2, 0) / allScores.length
  )
  const hi = mean + 0.5 * std
  const lo = mean - 0.5 * std
  if (score >= hi) return "好"
  if (score >= lo) return "中"
  return "差"
}

export function backendText(backend: InferenceBackend): string {
  return backend === "rust" ? "Rust" : "Python"
}

export function backendLabel(backend: InferenceBackend, device: string): string {
  if (backend === "python") return "Python"
  return device === "cuda" ? "Rust · CUDA" : "Rust · CPU"
}

export function averageScore(results: ScoreResult[]): number | null {
  if (!results.length) {
    return null
  }
  return results.reduce((sum, item) => sum + item.quality_score, 0) / results.length
}

export function formatScore(value: number | null | undefined, digits = 2): string {
  if (value == null || Number.isNaN(value)) {
    return "—"
  }
  return value.toFixed(digits)
}

export function clampPercentage(value: number): number {
  return Math.max(0, Math.min(100, value))
}
