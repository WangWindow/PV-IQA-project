import type { JobSummary, ScoreResult } from "@/lib/types"

export type JobTrendPoint = {
  id: string
  label: string
  score: number
  backend: JobSummary["backend"]
  status: JobSummary["status"]
  createdAt: string
}

export type ScoreBandPoint = {
  name: string
  count: number
}

export type RankedScorePoint = {
  rank: number
  score: number
  label: string
}

const QUALITY_BANDS = [
  { name: "好", min: 65 },
  { name: "中", min: 35 },
  { name: "差", min: 0 },
] as const

export function scoreSpread(results: ScoreResult[]): number | null {
  if (!results.length) {
    return null
  }
  const highest = results[0]?.quality_score ?? null
  const lowest = results.at(-1)?.quality_score ?? null
  if (highest == null || lowest == null) {
    return null
  }
  return highest - lowest
}

export function buildJobTrendData(jobs: JobSummary[], limit = 8): JobTrendPoint[] {
  return jobs
    .filter((job) => job.average_score != null)
    .slice(0, limit)
    .reverse()
    .map((job, index) => ({
      id: job.id,
      label:
        new Date(job.created_at).toLocaleTimeString("zh-CN", {
          hour: "2-digit",
          minute: "2-digit",
        }) || `任务 ${index + 1}`,
      score: job.average_score ?? 0,
      backend: job.backend,
      status: job.status,
      createdAt: job.created_at,
    }))
}

export function buildScoreDistribution(results: ScoreResult[]): ScoreBandPoint[] {
  return QUALITY_BANDS.map((band, index) => {
    const upperBound = QUALITY_BANDS[index - 1]?.min ?? Number.POSITIVE_INFINITY
    const count = results.filter(
      (result) => result.quality_score >= band.min && result.quality_score < upperBound
    ).length
    return {
      name: band.name,
      count,
    }
  })
}

export function buildRankedScoreData(results: ScoreResult[], limit = 24): RankedScorePoint[] {
  return results.slice(0, limit).map((result, index) => ({
    rank: index + 1,
    score: result.quality_score,
    label: result.relative_path,
  }))
}
