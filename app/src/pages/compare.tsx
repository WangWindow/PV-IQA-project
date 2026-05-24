import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { ArrowLeftRight, ChevronDown, Columns2 } from "lucide-react"

import type { JobRecord, ModelMeta, ScoreResult } from "@/lib/types"
import { buildRankedScoreData, buildScoreDistribution } from "@/lib/analytics"
import { createComparison, fetchJob, fetchJobs, fetchModels } from "@/lib/api"
import { formatScore, formatTime, qualityLabel } from "@/lib/format"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { StatCard } from "@/components/stat-card"
import { ScoreDistributionChart, RankedScoreChart } from "@/components/charts"
import { ImagePreview, type PreviewImage } from "@/components/image-preview"
import { Spinner } from "@/components/ui/spinner"

function useSearchParams() {
  const get = (key: string) => new URLSearchParams(window.location.search).get(key)
  const set = (params: Record<string, string>) => {
    const sp = new URLSearchParams(window.location.search)
    for (const [k, v] of Object.entries(params)) { if (v) sp.set(k, v); else sp.delete(k) }
    window.history.replaceState(null, "", sp.toString() ? "?" + sp.toString() : window.location.pathname)
  }
  return { get, set }
}

function ModelMetaBox({ model }: { model: ModelMeta | undefined }) {
  if (!model) return <div />
  return (
    <details className="rounded-lg border p-3 text-sm">
      <summary className="cursor-pointer font-medium flex items-center gap-2 select-none">
        <ChevronDown className="size-3 transition-transform group-open:rotate-180" />
        {model.name || model.run_name}
      </summary>
      <div className="mt-2 space-y-1">
        <div><span className="text-muted-foreground">运行名:</span> <code className="text-xs">{model.run_name}</code></div>
        {model.duration && <div><span className="text-muted-foreground">训练时长:</span> {model.duration}</div>}
        {model.metrics?.eer_aoc && <div><span className="text-muted-foreground">EER-AOC:</span> <strong>{model.metrics.eer_aoc}</strong></div>}
        {model.metrics?.err_roi_auc && <div><span className="text-muted-foreground">err_roi AUC:</span> <strong>{model.metrics.err_roi_auc}</strong></div>}
        {model.params && Object.keys(model.params).length > 0 && (
          <details className="mt-1">
            <summary className="cursor-pointer text-xs text-muted-foreground">超参数</summary>
            <div className="mt-1 grid grid-cols-3 gap-x-2 gap-y-0.5 text-xs">
              {Object.entries(model.params).map(([k, v]) => (
                <div key={k} className="contents"><span className="text-muted-foreground">{k}</span><span className="col-span-2">{v}</span></div>
              ))}
            </div>
          </details>
        )}
      </div>
    </details>
  )
}

function JobResultPanel({ job }: { job: JobRecord }) {
  const allScores = job.results.map((r) => r.quality_score)
  const distribution = useMemo(() => buildScoreDistribution(job.results), [job.results])
  const rankedData = useMemo(() => buildRankedScoreData(job.results), [job.results])
  const [preview, setPreview] = useState<PreviewImage | null>(null)
  const topResults = job.results.slice(0, 6)

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-3 gap-2">
        <StatCard label="平均分" value={formatScore(job.average_score ?? 0)} />
        <StatCard label="最高分" value={formatScore(job.best_score ?? 0)} />
        <StatCard label="图片数" value={String(job.result_count)} />
      </div>
      {distribution.length > 0 && <ScoreDistributionChart data={distribution} height={100} />}
      {rankedData.length > 0 && <RankedScoreChart data={rankedData} height={100} />}

      <div className="grid grid-cols-3 gap-2">
        {topResults.map((r) => (
          <button key={r.id} type="button"
            onClick={() => setPreview({ src: r.public_url, alt: r.relative_path, caption: `${r.relative_path} · ${formatScore(r.quality_score)}` })}
            className="overflow-hidden rounded-lg border bg-card text-left transition-all hover:border-primary/50">
            <img src={r.public_url} alt={r.relative_path} loading="lazy" className="h-20 w-full object-cover" />
            <div className="p-1.5 flex items-center justify-between">
              <span className="text-xs font-medium">{formatScore(r.quality_score)}</span>
              <Badge variant="secondary" className="text-[10px]">{qualityLabel(r.quality_score, allScores)}</Badge>
            </div>
          </button>
        ))}
      </div>

      <div className="space-y-0.5">
        {job.results.map((r, i) => (
          <div key={r.id ?? i} className="flex items-center gap-2 text-xs rounded px-1 py-0.5 hover:bg-muted/50">
            <span className="w-5 text-right text-muted-foreground tabular-nums">{i + 1}</span>
            <span className="flex-1 truncate font-mono">{r.relative_path}</span>
            <span className="w-12 text-right tabular-nums">{formatScore(r.quality_score)}</span>
            <Badge variant="outline" className="text-[10px]">{qualityLabel(r.quality_score, allScores)}</Badge>
          </div>
        ))}
      </div>

      {preview && <ImagePreview image={preview} onOpenChange={(open) => !open && setPreview(null)} />}
    </div>
  )
}

function PerImageCompareDialog({ open, onClose, leftJob, rightJob }: {
  open: boolean; onClose: () => void; leftJob: JobRecord; rightJob: JobRecord
}) {
  const [preview, setPreview] = useState<PreviewImage | null>(null)
  const leftMap = useMemo(() => { const m = new Map<string, ScoreResult>(); leftJob.results.forEach((r) => m.set(r.relative_path, r)); return m }, [leftJob.results])
  const rightMap = useMemo(() => { const m = new Map<string, ScoreResult>(); rightJob.results.forEach((r) => m.set(r.relative_path, r)); return m }, [rightJob.results])
  const allPaths = Array.from(new Set([...leftMap.keys(), ...rightMap.keys()])).sort()
  const lScores = leftJob.results.map((r) => r.quality_score)
  const rScores = rightJob.results.map((r) => r.quality_score)

  return (
    <Dialog open={open} onOpenChange={(o) => { if (!o) onClose() }}>
      <DialogContent className="max-w-4xl max-h-[85vh] overflow-hidden flex flex-col">
        <DialogHeader><DialogTitle>逐张图片对比</DialogTitle></DialogHeader>
        <div className="flex-1 overflow-y-auto custom-scrollbar space-y-1">
          {allPaths.map((path) => {
            const left = leftMap.get(path); const right = rightMap.get(path)
            const imgUrl = left?.public_url || right?.public_url || ""
            return (
              <div key={path} className="grid grid-cols-[1fr_auto_1fr] gap-3 items-center rounded-lg border p-2 hover:bg-muted/30">
                <div className="text-right">
                  {left ? <><span className="text-sm font-medium tabular-nums">{formatScore(left.quality_score)}</span><Badge variant="secondary" className="ml-1 text-[10px]">{qualityLabel(left.quality_score, lScores)}</Badge></> : <span className="text-xs text-muted-foreground">—</span>}
                </div>
                <button type="button" onClick={() => imgUrl && setPreview({ src: imgUrl, alt: path, caption: path })} className="shrink-0 overflow-hidden rounded border hover:ring-1 ring-primary/50 transition-all">
                  {imgUrl ? <img src={imgUrl} alt={path} loading="lazy" className="h-14 w-20 object-cover" /> : <div className="h-14 w-20 bg-muted flex items-center justify-center text-[10px] text-muted-foreground">无图</div>}
                </button>
                <div className="text-left">
                  {right ? <><span className="text-sm font-medium tabular-nums">{formatScore(right.quality_score)}</span><Badge variant="secondary" className="ml-1 text-[10px]">{qualityLabel(right.quality_score, rScores)}</Badge></> : <span className="text-xs text-muted-foreground">—</span>}
                </div>
              </div>
            )
          })}
        </div>
        {preview && <ImagePreview image={preview} onOpenChange={(open) => !open && setPreview(null)} />}
      </DialogContent>
    </Dialog>
  )
}

export function ComparePage() {
  const sp = useSearchParams()
  const [models, setModels] = useState<ModelMeta[]>([])
  const [jobs, setJobs] = useState<JobRecord[]>([])
  const [loading, setLoading] = useState(true)
  const [leftJobId, setLeftJobId] = useState(sp.get("left") || "")
  const [leftJob, setLeftJob] = useState<JobRecord | null>(null)
  const [rightJobId, setRightJobId] = useState(sp.get("right") || "")
  const [rightJob, setRightJob] = useState<JobRecord | null>(null)
  const [rightModel, setRightModel] = useState("")
  const [comparing, setComparing] = useState(false)
  const [detailOpen, setDetailOpen] = useState(false)

  const leftRef = useRef<HTMLDivElement>(null)
  const rightRef = useRef<HTMLDivElement>(null)
  const syncingRef = useRef(false)

  useEffect(() => {
    (async () => {
      const [m, summaries] = await Promise.all([fetchModels(), fetchJobs()])
      setModels(m)
      const completed = summaries.filter((j) => j.status === "completed").slice(0, 30)
      let records: JobRecord[] = []
      if (completed.length > 0) { records = await Promise.all(completed.map((j) => fetchJob(j.id))); setJobs(records) }
      const lid = sp.get("left")
      if (lid) { const lj = records.find((j) => j.id === lid) || await fetchJob(lid).catch(() => null); if (lj) setLeftJob(lj) }
      const rid = sp.get("right")
      if (rid) { const rj = await fetchJob(rid).catch(() => null); if (rj) setRightJob(rj) }
      setLoading(false)
    })().catch(() => setLoading(false))
  }, [])

  const handleSelectJob = (id: string) => {
    if (!id) return
    setLeftJobId(id); setLeftJob(jobs.find((j) => j.id === id) ?? null)
    setRightJob(null); setRightJobId(""); setRightModel("")
    sp.set({ left: id, right: "" })
  }

  const handleCompare = useCallback(async () => {
    if (!leftJobId || !rightModel) return
    setComparing(true)
    try {
      const { new_job } = await createComparison(leftJobId, rightModel)
      setRightJob(new_job); setRightJobId(new_job.id)
      sp.set({ left: leftJobId, right: new_job.id })
      const poll = setInterval(async () => {
        const updated = await fetchJob(new_job.id)
        setRightJob(updated)
        if (updated.status !== "running") { clearInterval(poll); setComparing(false) }
      }, 1500)
    } catch { setComparing(false) }
  }, [leftJobId, rightModel])

  const handleScroll = (source: "left" | "right") => {
    if (syncingRef.current) return
    syncingRef.current = true
    const src = source === "left" ? leftRef.current : rightRef.current
    const dst = source === "left" ? rightRef.current : leftRef.current
    if (src && dst) dst.scrollTop = src.scrollTop
    requestAnimationFrame(() => { syncingRef.current = false })
  }

  if (loading) return <div className="flex min-h-[50vh] items-center justify-center"><Spinner className="size-5" /></div>

  const leftMeta = leftJob ? models.find((m) => m.run_name === leftJob.run_name) : undefined
  const rightMeta = rightJob ? models.find((m) => m.run_name === rightJob.run_name) : undefined
  const bothReady = leftJob && rightJob && rightJob.status === "completed"

  return (
    <div className="space-y-4 p-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">模型对比</h1>
          <p className="mt-1 text-sm text-muted-foreground">选择一个已完成的评估任务作为基准，再用不同模型对比评分结果</p>
        </div>
        <div className="flex items-center gap-2">
          <Button onClick={handleCompare} disabled={!rightModel || comparing}>
            {comparing ? <><Spinner className="mr-2 size-3" />评分中...</> : <><ArrowLeftRight className="mr-2 size-4" />开始对比</>}
          </Button>
          {bothReady && (
            <Button variant="outline" onClick={() => setDetailOpen(true)}>
              <Columns2 className="mr-2 size-4" />逐张对比
            </Button>
          )}
        </div>
      </div>

      <div className="grid grid-cols-2 gap-6" style={{ height: "calc(100vh - 9rem)" }}>
        <Card className="flex flex-col overflow-hidden">
          <CardHeader className="pb-3 shrink-0"><CardTitle className="text-sm">基准模型</CardTitle></CardHeader>
          <CardContent className="flex-1 overflow-hidden grid grid-rows-[auto_auto_1fr] gap-3 pb-6">
            <select className="w-full rounded-lg border px-3 py-2 text-sm" value={leftJobId} onChange={(e) => handleSelectJob(e.target.value)}>
              <option value="">选择已完成的任务...</option>
              {jobs.map((j) => (<option key={j.id} value={j.id}>{j.run_name} — {formatTime(j.created_at)}</option>))}
            </select>

            <div><ModelMetaBox model={leftMeta} /></div>

            <div ref={leftRef} onScroll={() => handleScroll("left")} className="overflow-y-auto custom-scrollbar">
              {leftJob ? <JobResultPanel job={leftJob} /> : <p className="py-8 text-center text-sm text-muted-foreground">选择任务以查看结果</p>}
            </div>
          </CardContent>
        </Card>

        <Card className="flex flex-col overflow-hidden">
          <CardHeader className="pb-3 shrink-0"><CardTitle className="text-sm">对比模型</CardTitle></CardHeader>
          <CardContent className="flex-1 overflow-hidden grid grid-rows-[auto_auto_1fr] gap-3 pb-6">
            <select className="w-full rounded-lg border px-3 py-2 text-sm" value={rightModel} onChange={(e) => setRightModel(e.target.value)} disabled={!leftJobId}>
              <option value="">选择对比模型...</option>
              {models.filter((m) => m.run_name !== leftJob?.run_name).map((m) => (
                <option key={m.run_name} value={m.run_name}>{m.name || m.run_name}</option>
              ))}
            </select>

            <div><ModelMetaBox model={rightMeta} /></div>

            <div ref={rightRef} onScroll={() => handleScroll("right")} className="overflow-y-auto custom-scrollbar">
              {rightJob ? (
                <div className="space-y-2">
                  {rightJob.status === "running" && <div className="flex items-center gap-2 text-sm"><Spinner className="size-3" /> 正在评分...</div>}
                  {rightJob.status === "completed" && <JobResultPanel job={rightJob} />}
                </div>
              ) : (
                <p className="py-8 text-center text-sm text-muted-foreground">选择基准任务和对比模型后<br/>点击「开始对比」</p>
              )}
            </div>
          </CardContent>
        </Card>
      </div>

      {bothReady && leftJob && <PerImageCompareDialog open={detailOpen} onClose={() => setDetailOpen(false)} leftJob={leftJob} rightJob={rightJob!} />}
    </div>
  )
}
