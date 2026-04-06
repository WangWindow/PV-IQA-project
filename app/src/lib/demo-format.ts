import type { JobStatus, ScoreResult } from "@/lib/types"

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

export function averageScore(results: ScoreResult[]): number | null {
  if (!results.length) {
    return null
  }
  return results.reduce((sum, item) => sum + item.quality_score, 0) / results.length
}

export function clampPercentage(value: number): number {
  return Math.max(0, Math.min(100, value))
}
