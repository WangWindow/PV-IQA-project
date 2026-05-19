import { useMemo, useState } from "react"
import { AnimatePresence, motion } from "motion/react"
import {
  ChartNoAxesCombined,
  CheckSquare,
  Download,
  Eye,
  FileText,
  FolderClock,
  Image as ImageIcon,
  PauseCircle,
  Play,
  RotateCcw,
  Search,
  Square,
  Tag,
  Trash2,
  X,
} from "lucide-react"

import type { Dashboard } from "@/hooks/use-dashboard"
import type { JobRecord } from "@/lib/types"
import type { JobStatus } from "@/lib/types"
import { buildJobTrendData, buildRankedScoreData, buildScoreDistribution } from "@/lib/analytics"
import {
  backendText,
  clampPercentage,
  formatScore,
  formatTime,
  qualityLabel,
  statusText,
} from "@/lib/format"
import { cn, downloadCsv } from "@/lib/utils"
import { batchDeleteJobs, batchRerunJobs, updateJobTags } from "@/lib/api"
import { Disclosure } from "@/components/disclosure"
import { ImagePreview, type PreviewImage } from "@/components/image-preview"
import { StatCard } from "@/components/stat-card"
import {
  JobTrendChart,
  RankedScoreChart,
  ScoreDistributionChart,
} from "@/components/charts"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import {
  Card,
  CardAction,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import {
  Empty,
  EmptyDescription,
  EmptyHeader,
  EmptyMedia,
  EmptyTitle,
} from "@/components/ui/empty"
import { Progress } from "@/components/ui/progress"
import { ScrollArea } from "@/components/ui/scroll-area"
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet"
import { Spinner } from "@/components/ui/spinner"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"

/* ───────────────────────────────────────────
   Types & Helpers
   ─────────────────────────────────────────── */

type HistoryFilter = "all" | "running" | "interrupted" | "completed" | "failed"

const STATUS_BORDER: Record<string, string> = {
  running: "border-l-[var(--primary)]",
  completed: "border-l-[var(--chart-2)]",
  interrupted: "border-l-[var(--chart-4)]",
  failed: "border-l-[var(--destructive)]",
}

const STATUS_BADGE_BG: Record<string, string> = {
  running: "bg-[var(--primary)] text-[var(--primary-foreground)] animate-infrared-pulse",
  completed: "bg-[var(--chart-2)] text-[var(--primary-foreground)]",
  interrupted: "bg-[var(--chart-4)] text-[var(--primary-foreground)]",
  failed: "bg-[var(--destructive)] text-[var(--destructive-foreground)]",
}

const FILTER_STATUS_KEYS: ReadonlyArray<HistoryFilter> = [
  "running",
  "interrupted",
  "completed",
  "failed",
] as const

/* ───────────────────────────────────────────
   HistoryFilterControls
   ─────────────────────────────────────────── */

function HistoryFilterControls({
  filter,
  setFilter,
  className,
}: {
  filter: HistoryFilter
  setFilter: (value: HistoryFilter) => void
  className?: string
}) {
  const filters = [
    { value: "all" as const, label: "全部" },
    { value: "running" as const, label: "处理中" },
    { value: "interrupted" as const, label: "中断" },
    { value: "completed" as const, label: "已完成" },
    { value: "failed" as const, label: "失败" },
  ]

  return (
    <ToggleGroup
      type="single"
      variant="outline"
      value={filter}
      className={className}
      onValueChange={(value) => {
        if (value === "all" || (FILTER_STATUS_KEYS as ReadonlyArray<string>).includes(value)) {
          setFilter(value as HistoryFilter)
        }
      }}
    >
      {filters.map(({ value, label }) => (
        <ToggleGroupItem
          key={value}
          value={value}
          aria-label={`显示${label}任务`}
          className={cn(
            "relative px-3 py-1.5 text-xs font-medium transition-all duration-200",
            "data-[state=on]:bg-primary data-[state=on]:text-primary-foreground data-[state=on]:infrared-glow",
            "data-[state=off]:text-muted-foreground data-[state=off]:hover:bg-accent/40"
          )}
        >
          {label}
        </ToggleGroupItem>
      ))}
    </ToggleGroup>
  )
}

/* ───────────────────────────────────────────
   StatusBadge – unified status pill
   ─────────────────────────────────────────── */

function StatusBadge({ status }: { status: string }) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-semibold shadow-sm transition-all duration-300",
        STATUS_BADGE_BG[status as JobStatus] ?? "bg-muted text-muted-foreground"
      )}
    >
      {statusText(status as JobStatus)}
    </span>
  )
}

/* ───────────────────────────────────────────
   JobList – card rows with left accent
   ─────────────────────────────────────────── */

function JobList({
  dashboard,
  jobs,
  onPick,
  batchMode,
  selectedIds,
  onToggleSelect,
}: {
  dashboard: Dashboard
  jobs: Dashboard["jobs"]
  onPick?: () => void
  batchMode?: boolean
  selectedIds?: Set<string>
  onToggleSelect?: (id: string) => void
}) {
  return (
    <div className="flex flex-col gap-2.5">
      {jobs.map((job, idx) => {
        const accentColor = STATUS_BORDER[job.status] ?? "border-l-transparent"
        const isSelected = selectedIds?.has(job.id)
        return (
          <div
            key={job.id}
            className={cn(
              "group relative flex flex-col gap-2.5 rounded-xl border border-border bg-card p-4 text-left",
              "border-l-[3px]",
              accentColor,
              "transition-all duration-200 ease-out",
              "hover:border-primary/50 hover:scale-[1.01]",
              "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
              "animate-fade-in-up",
              dashboard.selectedJob?.id === job.id && "bg-accent/20 border-primary",
            )}
            style={{ animationDelay: `${idx * 50}ms` }}
          >
            <div className="flex items-start gap-3">
              {batchMode ? (
                <input
                  type="checkbox"
                  checked={isSelected}
                  onChange={() => onToggleSelect?.(job.id)}
                  className="mt-0.5 size-4 shrink-0 accent-primary"
                  onClick={(e) => e.stopPropagation()}
                />
              ) : null}
              <button
                type="button"
                onClick={() => {
                  void dashboard.selectJob(job.id)
                  onPick?.()
                }}
                className="flex-1 min-w-0 text-left"
              >
                <div className="flex items-center justify-between gap-2">
                  <div className="flex flex-wrap items-center gap-1.5">
                    <Badge variant="outline" className="text-[10px] px-1.5 py-0">
                      {job.kind === "image" ? "单图" : "文件夹"}
                    </Badge>
                    <StatusBadge status={job.status} />
                  </div>
                  <div className="stat-value text-sm tabular-nums text-muted-foreground group-hover:text-primary transition-colors">
                    {formatScore(job.average_score)}
                  </div>
                </div>
                <div className="min-w-0 mt-2">
                  <div className="truncate text-sm font-semibold">{job.stage}</div>
                  <div className="mt-1 font-data text-xs text-muted-foreground">
                    {formatTime(job.created_at)}
                  </div>
                </div>
                {job.status === "running" ? (
                  <Progress value={clampPercentage(job.progress)} className="h-1.5 mt-2" />
                ) : (
                  <div className="font-data text-xs text-muted-foreground mt-2">
                    {job.result_count} 条结果
                  </div>
                )}
              </button>
            </div>
          </div>
        )
      })}
    </div>
  )
}

/* ───────────────────────────────────────────
   DetailEmpty
   ─────────────────────────────────────────── */

function DetailEmpty() {
  return (
    <Empty className="border bg-muted/5 animate-fade-in-up">
      <EmptyHeader>
        <EmptyMedia variant="icon">
          <FolderClock aria-hidden="true" className="text-primary/60" />
        </EmptyMedia>
        <EmptyTitle>选择一条任务</EmptyTitle>
        <EmptyDescription>
          从左侧列表中选择任务后，这里显示详细结果与可选分析。
        </EmptyDescription>
      </EmptyHeader>
    </Empty>
  )
}

/* ───────────────────────────────────────────
   JobMetaLine
   ─────────────────────────────────────────── */

function JobMetaLine({ job }: { job: JobRecord }) {
  const meta = [
    { label: "Run", value: job.run_name },
    { label: "创建", value: formatTime(job.created_at) },
    { label: "更新", value: formatTime(job.updated_at) },
    ...(job.completed_at ? [{ label: "完成", value: formatTime(job.completed_at) }] : []),
    { label: "已处理", value: `${job.processed_count}/${job.input_count}` },
  ]

  return (
    <div className="flex flex-wrap gap-x-5 gap-y-1">
      {meta.map(({ label, value }) => (
        <span key={label} className="inline-flex items-baseline gap-1.5 font-data text-xs text-muted-foreground">
          <span className="text-[10px] uppercase tracking-wider text-muted-foreground/60">
            {label}
          </span>
          {value}
        </span>
      ))}
    </div>
  )
}

/* ───────────────────────────────────────────
   ActionButton – consistent action trigger
   ─────────────────────────────────────────── */

function ActionButton({
  icon: Icon,
  label,
  variant = "outline",
  infrared = false,
  ...rest
}: {
  icon: React.ComponentType<{ className?: string; "aria-hidden"?: boolean; "data-icon"?: string }>
  label: string
  variant?: "outline" | "ghost" | "destructive"
  infrared?: boolean
} & React.ComponentProps<typeof Button>) {
  return (
    <Button
      size="sm"
      variant={variant}
      className={cn(
        "transition-all duration-200",
        infrared && "hover:text-primary"
      )}
      {...rest}
    >
      <Icon aria-hidden data-icon="inline-start" />
      {label}
    </Button>
  )
}

/* ───────────────────────────────────────────
   RunningWidget
   ─────────────────────────────────────────── */

function RunningWidget({ job }: { job: JobRecord }) {
  return (
    <div className="rounded-xl border border-border bg-accent/5 p-4 animate-fade-in-up">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-center gap-3">
          <Spinner />
          <div>
            <div className="font-semibold">{job.stage}</div>
            <div className="font-data text-sm text-muted-foreground">
              {Math.round(job.progress)}%
            </div>
          </div>
        </div>
        <div className="font-data text-sm text-primary tabular-nums">
          进度会持续刷新
        </div>
      </div>
      <Progress value={clampPercentage(job.progress)} className="mt-4 h-1.5" />
    </div>
  )
}

/* ───────────────────────────────────────────
   ErrorAlert – unified error/interrupted
   ─────────────────────────────────────────── */

function ErrorAlert({ job }: { job: JobRecord }) {
  const isFailed = job.status === "failed"
  return (
    <Alert variant={isFailed ? "destructive" : "default"} className="animate-fade-in-up">
      <AlertTitle>{isFailed ? "任务失败" : "任务已中断"}</AlertTitle>
      <AlertDescription>{job.error}</AlertDescription>
    </Alert>
  )
}

/* ───────────────────────────────────────────
   historyPage – main export
   ─────────────────────────────────────────── */

export function HistoryPage({ dashboard }: { dashboard: Dashboard }) {
  const [filter, setFilter] = useState<HistoryFilter>("all")
  const [searchQuery, setSearchQuery] = useState("")
  const [deleteTargetId, setDeleteTargetId] = useState<string | null>(null)
  const [mobileSheetOpen, setMobileSheetOpen] = useState(false)
  const [trendOpen, setTrendOpen] = useState(false)
  const [previewImage, setPreviewImage] = useState<PreviewImage | null>(null)
  const [selectedJobIds, setSelectedJobIds] = useState<Set<string>>(new Set())
  const [batchMode, setBatchMode] = useState(false)
  const [tagDialogOpen, setTagDialogOpen] = useState(false)
  const [batchTagValue, setBatchTagValue] = useState("")

  const filteredJobs = useMemo(() => {
    let result = dashboard.jobs

    if (filter !== "all") {
      result = result.filter((job) => job.status === filter)
    }

    const q = searchQuery.trim().toLowerCase()
    if (q) {
      result = result.filter(
        (job) =>
          job.run_name.toLowerCase().includes(q) ||
          job.id.toLowerCase().includes(q)
      )
    }

    return result
  }, [dashboard.jobs, filter, searchQuery])

  const selectedJob = dashboard.selectedJob
  const topResult = dashboard.topResult
  const deleteTarget = dashboard.jobs.find((job) => job.id === deleteTargetId) ?? null

  const trendData = useMemo(() => buildJobTrendData(dashboard.jobs, 10), [dashboard.jobs])

  const completedJobsWithScores = useMemo(
    () => dashboard.jobs.filter((job) => job.status === "completed" && job.average_score != null),
    [dashboard.jobs]
  )

  const overallAverageScore = useMemo(() => {
    if (!completedJobsWithScores.length) return null
    return (
      completedJobsWithScores.reduce((sum, job) => sum + (job.average_score ?? 0), 0) /
      completedJobsWithScores.length
    )
  }, [completedJobsWithScores])

  const selectedDistribution = useMemo(
    () => buildScoreDistribution(dashboard.selectedResults),
    [dashboard.selectedResults]
  )

  const selectedRankedScores = useMemo(
    () => buildRankedScoreData(dashboard.selectedResults),
    [dashboard.selectedResults]
  )

  const allScores = useMemo(
    () => dashboard.selectedResults.map((r) => r.quality_score),
    [dashboard.selectedResults]
  )

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -16 }}
      transition={{ duration: 0.22, ease: "easeOut" }}
      className="flex flex-col gap-6"
    >

      {dashboard.error ? (
        <Alert variant="destructive" className="animate-fade-in-up">
          <AlertTitle>操作失败</AlertTitle>
          <AlertDescription>{dashboard.error}</AlertDescription>
        </Alert>
      ) : null}
      <div className="flex items-center gap-3 pt-1">
        <span className="block h-5 w-1 rounded-full bg-primary" />
        <h2 className="text-lg font-bold tracking-tight">历史任务</h2>
        <span className="font-data text-xs text-muted-foreground">
          {dashboard.jobs.length} 任务 · {dashboard.activeRunningCount ?? 0} 处理中
        </span>
      </div>
      <div className="grid items-start gap-6 lg:grid-cols-[340px_minmax(0,1fr)]">

        <Card className="hidden lg:flex lg:sticky lg:top-5 lg:self-start animate-fade-in-up">
          <CardHeader>
            <CardTitle>任务列表</CardTitle>
            <CardAction className="flex flex-wrap items-center gap-2">
              <Badge variant="outline" className="font-data text-[10px]">
                {dashboard.jobs.length} 总任务
              </Badge>
              {dashboard.activeRunningCount ? (
                <Badge variant="outline" className="font-data text-[10px]">
                  {dashboard.activeRunningCount} 处理中
                </Badge>
              ) : null}
              <Button variant="ghost" size="sm" onClick={() => setTrendOpen(true)}>
                <ChartNoAxesCombined aria-hidden="true" data-icon="inline-start" />
                总体趋势
              </Button>
              <Button
                variant={batchMode ? "secondary" : "ghost"}
                size="sm"
                onClick={() => {
                  setBatchMode((v) => !v)
                  setSelectedJobIds(new Set())
                }}
              >
                {batchMode ? <CheckSquare className="size-3.5" /> : <Square className="size-3.5" />}
                批量操作
              </Button>
            </CardAction>
          </CardHeader>
          <CardContent className="flex flex-col gap-4">
            <div className="relative">
              <Search
                aria-hidden="true"
                className="absolute left-2.5 top-1/2 size-3.5 -translate-y-1/2 text-muted-foreground"
              />
              <Input
                type="search"
                placeholder="按名称或 ID 搜索…"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="h-7 pl-8 pr-7 text-xs"
              />
              {searchQuery ? (
                <button
                  type="button"
                  onClick={() => setSearchQuery("")}
                  className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors"
                >
                  <X aria-hidden="true" className="size-3" />
                </button>
              ) : null}
            </div>
            <HistoryFilterControls filter={filter} setFilter={setFilter} className="flex-wrap" />

            {batchMode && selectedJobIds.size > 0 ? (
              <div className="flex flex-wrap items-center gap-2 rounded-lg border border-primary/20 bg-primary/5 px-3 py-2">
                <span className="text-xs font-medium">已选 {selectedJobIds.size} 项</span>
                <div className="mx-1 h-3 w-px bg-border" />
                <button
                  type="button"
                  onClick={() => {
                    void batchRerunJobs(Array.from(selectedJobIds)).then(() => {
                      setSelectedJobIds(new Set())
                      void dashboard.refreshJobs()
                    })
                  }}
                  className="inline-flex items-center gap-1 rounded px-2 py-1 text-xs font-medium text-primary hover:bg-primary/10"
                >
                  <RotateCcw className="size-3" />
                  批量重跑
                </button>
                <button
                  type="button"
                  onClick={() => setTagDialogOpen(true)}
                  className="inline-flex items-center gap-1 rounded px-2 py-1 text-xs font-medium text-primary hover:bg-primary/10"
                >
                  <Tag className="size-3" />
                  批量标签
                </button>
                <button
                  type="button"
                  onClick={() => {
                    if (!confirm(`确定要删除选中的 ${selectedJobIds.size} 个任务吗？`)) return
                    void batchDeleteJobs(Array.from(selectedJobIds)).then(() => {
                      setSelectedJobIds(new Set())
                      void dashboard.refreshJobs()
                    })
                  }}
                  className="inline-flex items-center gap-1 rounded px-2 py-1 text-xs font-medium text-destructive hover:bg-destructive/10"
                >
                  <Trash2 className="size-3" />
                  批量删除
                </button>
                <button
                  type="button"
                  onClick={() => setSelectedJobIds(new Set())}
                  className="ml-auto inline-flex items-center gap-1 rounded px-2 py-1 text-xs text-muted-foreground hover:bg-muted"
                >
                  清空
                </button>
              </div>
            ) : null}

            {filteredJobs.length ? (
              <ScrollArea className="h-[calc(100vh-14rem)] custom-scrollbar pr-1">
                <JobList
                  dashboard={dashboard}
                  jobs={filteredJobs}
                  batchMode={batchMode}
                  selectedIds={selectedJobIds}
                  onToggleSelect={(id) =>
                    setSelectedJobIds((prev) => {
                      const next = new Set(prev)
                      if (next.has(id)) next.delete(id)
                      else next.add(id)
                      return next
                    })
                  }
                />
              </ScrollArea>
            ) : (
              <Empty className="border bg-muted/5 animate-fade-in-up">
                <EmptyHeader>
                  <EmptyTitle>没有匹配任务</EmptyTitle>
                  <EmptyDescription>切换筛选条件或先创建任务。</EmptyDescription>
                </EmptyHeader>
              </Empty>
            )}
          </CardContent>
        </Card>
        <Card className="animate-fade-in-up" style={{ animationDelay: "100ms" }}>
          <CardHeader>
            <CardTitle>任务详情</CardTitle>
            <CardAction className="flex flex-wrap items-center gap-2">
              <div className="md:hidden">
                <Sheet open={mobileSheetOpen} onOpenChange={setMobileSheetOpen}>
                  <SheetTrigger asChild>
                    <Button variant="outline" size="sm">
                      <Eye aria-hidden="true" data-icon="inline-start" />
                      任务列表
                    </Button>
                  </SheetTrigger>
                  <SheetContent
                    side="right"
                    className="scan-overlay"
                  >
                    <SheetHeader>
                      <SheetTitle>历史任务</SheetTitle>
                    </SheetHeader>
                    <div className="flex flex-col gap-4 px-4">
                      <div className="flex flex-wrap items-center gap-2">
                        <Badge variant="outline" className="font-data text-[10px]">
                          {dashboard.jobs.length} 总任务
                        </Badge>
                        {dashboard.activeRunningCount ? (
                          <Badge variant="outline" className="font-data text-[10px]">
                            {dashboard.activeRunningCount} 处理中
                          </Badge>
                        ) : null}
                        <Button variant="ghost" size="sm" onClick={() => setTrendOpen(true)}>
                          <ChartNoAxesCombined aria-hidden="true" data-icon="inline-start" />
                          总体趋势
                        </Button>
                      </div>
                      <div className="relative">
                        <Search
                          aria-hidden="true"
                          className="absolute left-2.5 top-1/2 size-3.5 -translate-y-1/2 text-muted-foreground"
                        />
                        <Input
                          type="search"
                          placeholder="按名称或 ID 搜索…"
                          value={searchQuery}
                          onChange={(e) => setSearchQuery(e.target.value)}
                          className="h-7 pl-8 pr-7 text-xs"
                        />
                        {searchQuery ? (
                          <button
                            type="button"
                            onClick={() => setSearchQuery("")}
                            className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors"
                          >
                            <X aria-hidden="true" className="size-3" />
                          </button>
                        ) : null}
                      </div>
                      <HistoryFilterControls filter={filter} setFilter={setFilter} className="flex-wrap" />
                      <JobList
                        dashboard={dashboard}
                        jobs={filteredJobs}
                        onPick={() => setMobileSheetOpen(false)}
                        batchMode={batchMode}
                        selectedIds={selectedJobIds}
                        onToggleSelect={(id) =>
                          setSelectedJobIds((prev) => {
                            const next = new Set(prev)
                            if (next.has(id)) next.delete(id)
                            else next.add(id)
                            return next
                          })
                        }
                      />
                    </div>
                  </SheetContent>
                </Sheet>
              </div>

              {selectedJob ? (
                <>
                  {selectedJob.status === "running" ? (
                    <ActionButton
                      icon={PauseCircle}
                      label="停止"
                      onClick={() => void dashboard.stopJob(selectedJob.id)}
                      disabled={
                        dashboard.mutatingJobId === selectedJob.id ||
                        dashboard.deletingJobId === selectedJob.id
                      }
                      infrared
                    />
                  ) : null}
                  {selectedJob.status === "interrupted" ? (
                    <ActionButton
                      icon={Play}
                      label="恢复"
                      onClick={() => void dashboard.resumeJob(selectedJob.id)}
                      disabled={
                        dashboard.mutatingJobId === selectedJob.id ||
                        dashboard.deletingJobId === selectedJob.id
                      }
                      infrared
                    />
                  ) : null}
                  {selectedJob.status === "completed" || selectedJob.status === "failed" ? (
                    <ActionButton
                      icon={RotateCcw}
                      label="重新运行"
                      onClick={() => void dashboard.rerunJob(selectedJob.id)}
                      disabled={
                        dashboard.mutatingJobId === selectedJob.id ||
                        dashboard.deletingJobId === selectedJob.id
                      }
                      infrared
                    />
                  ) : null}
                  {selectedJob && (
                    <ActionButton
                      icon={Download}
                      label="导出 CSV"
                      variant="ghost"
                      onClick={() => {
                        if (selectedJob) {
                          const headers = ["文件名", "相对路径", "质量分数", "评分等级"]
                          const rows = dashboard.selectedResults.map((r) => [
                            r.relative_path.split("/").pop() || r.relative_path,
                            r.relative_path,
                            r.quality_score.toFixed(2),
                            qualityLabel(r.quality_score, dashboard.selectedResults.map((sr) => sr.quality_score)),
                          ])
                          downloadCsv(`pv-iqa-${selectedJob.id.slice(0, 8)}.csv`, headers, rows)
                        }
                      }}
                    />
                  )}
                  {selectedJob && selectedJob.status === "completed" && (
                    <ActionButton
                      icon={FileText}
                      label="生成报告"
                      variant="ghost"
                      onClick={() => {
                        if (!selectedJob) return
                        const scores = dashboard.selectedResults.map((r) => r.quality_score)
                        const reportHtml = `
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>PV-IQA 质量评估报告 - ${selectedJob.run_name || selectedJob.id.slice(0, 8)}</title>
<style>
body{font-family:system-ui,-apple-system,sans-serif;max-width:900px;margin:40px auto;padding:20px;line-height:1.6;color:#333}
h1{border-bottom:2px solid #0066cc;padding-bottom:10px;color:#0066cc}
.meta{display:grid;grid-template-columns:repeat(2,1fr);gap:12px;margin:20px 0;padding:16px;background:#f5f7fa;border-radius:8px}
.meta-item{display:flex;justify-content:space-between;font-size:14px}
.meta-label{color:#666}.meta-value{font-weight:600}
table{width:100%;border-collapse:collapse;margin:20px 0;font-size:13px}
th,td{padding:10px 12px;text-align:left;border-bottom:1px solid #e5e7eb}
th{background:#f8fafc;font-weight:600;color:#374151}
tr:hover{background:#f9fafb}
.score-bar{height:6px;border-radius:3px;background:#e5e7eb;overflow:hidden}
.score-bar>div{height:100%;border-radius:3px;background:linear-gradient(90deg,#ef4444,#f59e0b,#22c55e)}
.footer{margin-top:40px;padding-top:20px;border-top:1px solid #e5e7eb;font-size:12px;color:#9ca3af;text-align:center}
@media print{body{margin:0;padding:20px}}
</style>
</head>
<body>
<h1>掌静脉图像质量评估报告</h1>
<div class="meta">
  <div class="meta-item"><span class="meta-label">任务名称</span><span class="meta-value">${selectedJob.run_name || selectedJob.id.slice(0, 8)}</span></div>
  <div class="meta-item"><span class="meta-label">任务类型</span><span class="meta-value">${selectedJob.kind === "image" ? "单图评分" : "批量评分"}</span></div>
  <div class="meta-item"><span class="meta-label">图片数量</span><span class="meta-value">${selectedJob.result_count}</span></div>
  <div class="meta-item"><span class="meta-label">平均得分</span><span class="meta-value">${selectedJob.average_score?.toFixed(2) ?? "—"}</span></div>
  <div class="meta-item"><span class="meta-label">最高分</span><span class="meta-value">${selectedJob.best_score?.toFixed(2) ?? "—"}</span></div>
  <div class="meta-item"><span class="meta-label">最低分</span><span class="meta-value">${selectedJob.worst_score?.toFixed(2) ?? "—"}</span></div>
  <div class="meta-item"><span class="meta-label">创建时间</span><span class="meta-value">${formatTime(selectedJob.created_at)}</span></div>
  <div class="meta-item"><span class="meta-label">推理后端</span><span class="meta-value">${selectedJob.backend}</span></div>
</div>
<h2>评分结果明细</h2>
<table>
<thead><tr><th>排名</th><th>文件名</th><th>质量分数</th><th>等级</th><th>得分分布</th></tr></thead>
<tbody>
${dashboard.selectedResults.map((r, i) => {
  const pct = scores.length > 0 ? ((r.quality_score - Math.min(...scores)) / (Math.max(...scores) - Math.min(...scores) || 1)) * 100 : 50
  return `<tr><td>${i + 1}</td><td>${r.relative_path.split("/").pop() || r.relative_path}</td><td>${r.quality_score.toFixed(2)}</td><td>${qualityLabel(r.quality_score, scores)}</td><td><div class="score-bar"><div style="width:${pct}%"></div></div></td></tr>`
}).join("")}
</tbody>
</table>
<div class="footer">由 PV-IQA 掌静脉图像质量评估系统生成 · ${new Date().toLocaleString("zh-CN")}</div>
</body>
</html>`
                        const w = window.open("", "_blank")
                        if (w) {
                          w.document.write(reportHtml)
                          w.document.close()
                        }
                      }}
                    />
                  )}
                  <ActionButton
                    icon={Trash2}
                    label="删除"
                    variant="ghost"
                    onClick={() => setDeleteTargetId(selectedJob.id)}
                    disabled={
                      dashboard.mutatingJobId === selectedJob.id ||
                      dashboard.deletingJobId === selectedJob.id
                    }
                  />
                </>
              ) : null}
            </CardAction>
          </CardHeader>

          <CardContent>
            <AnimatePresence mode="wait">
              {selectedJob ? (
                <motion.div
                  key={selectedJob.id}
                  initial={{ opacity: 0, y: 12 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -12 }}
                  transition={{ duration: 0.25, ease: "easeOut" }}
                  className="flex flex-col gap-5"
                >
                  <div className="flex flex-wrap items-center gap-2">
                    <StatusBadge status={selectedJob.status} />
                    <Badge variant="outline" className="font-data text-[10px]">
                      {selectedJob.kind === "image" ? "单图" : "文件夹"}
                    </Badge>
                    <Badge variant="outline" className="font-data text-[10px]">
                      {backendText(selectedJob.backend)}
                    </Badge>
                  </div>

                  <JobMetaLine job={selectedJob} />

                  {selectedJob.status === "running" ? (
                    <RunningWidget job={selectedJob} />
                  ) : null}

                  {selectedJob.error ? <ErrorAlert job={selectedJob} /> : null}
                  {selectedJob.kind === "image" && topResult ? (
                    <div className="grid gap-6 lg:grid-cols-[minmax(0,300px)_minmax(0,1fr)]">
                      <button
                        type="button"
                        onClick={() =>
                          setPreviewImage({
                            src: topResult.public_url,
                            alt: topResult.relative_path,
                            caption: `${topResult.relative_path} · ${formatScore(topResult.quality_score)}`,
                          })
                        }
                        className="group relative overflow-hidden rounded-xl border border-border transition-all duration-300 hover:border-primary/40 hover:shadow-md focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                      >
                        <div className="scan-overlay absolute inset-0 z-10 rounded-xl" />
                        <img
                          src={topResult.public_url}
                          alt={topResult.relative_path}
                          width={360}
                          height={360}
                          className="h-72 w-full object-cover transition duration-300 group-hover:scale-[1.02]"
                        />
                      </button>
                      <div className="flex flex-col gap-4">
                        <div>
                          <div className="text-sm text-muted-foreground">质量分数</div>
                          <div className="mt-2 flex items-center gap-3">
                            <div className="stat-value text-4xl">{formatScore(topResult.quality_score)}</div>
                            <Badge variant="secondary">
                              {qualityLabel(topResult.quality_score, allScores)}
                            </Badge>
                          </div>
                        </div>
                        <Progress
                          value={clampPercentage(topResult.quality_score)}
                          className="h-1.5"
                        />
                        <div className="rounded-xl border border-border bg-accent/5 p-4">
                          <div className="text-sm text-muted-foreground">文件路径</div>
                          <div className="mt-2 break-all font-data text-sm font-medium">
                            {topResult.relative_path}
                          </div>
                        </div>
                      </div>
                    </div>
                  ) : null}
                  {selectedJob.kind === "folder" ? (
                    <div className="flex flex-col gap-5">
                      <div className="grid gap-3 md:grid-cols-3">
                        <StatCard label="平均分" value={formatScore(selectedJob.average_score)} />
                        <StatCard label="最高分" value={formatScore(selectedJob.best_score)} />
                        <StatCard label="结果数量" value={String(dashboard.selectedResults.length)} />
                      </div>

                      {dashboard.selectedResults.length ? (
                        <ScrollArea className="h-105 rounded-xl border border-border custom-scrollbar">
                          <Table>
                            <TableHeader>
                              <TableRow>
                                <TableHead>预览</TableHead>
                                <TableHead>路径</TableHead>
                                <TableHead>等级</TableHead>
                                <TableHead className="text-right">分数</TableHead>
                              </TableRow>
                            </TableHeader>
                            <TableBody>
                              {dashboard.selectedResults.map((result) => (
                                <TableRow key={result.id} className="group transition-colors hover:bg-accent/10">
                                  <TableCell>
                                    <button
                                      type="button"
                                      onClick={() =>
                                        setPreviewImage({
                                          src: result.public_url,
                                          alt: result.relative_path,
                                          caption: `${result.relative_path} · ${formatScore(result.quality_score)}`,
                                        })
                                      }
                                      className="overflow-hidden rounded-lg border border-border transition-all duration-200 hover:border-primary/30 hover:shadow-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                                    >
                                      <img
                                        src={result.public_url}
                                        alt={result.relative_path}
                                        width={64}
                                        height={48}
                                        loading="lazy"
                                        className="h-12 w-16 object-cover transition duration-200 group-hover:scale-[1.04]"
                                      />
                                    </button>
                                  </TableCell>
                                  <TableCell className="max-w-[320px] truncate font-data text-xs">
                                    {result.relative_path}
                                  </TableCell>
                                  <TableCell>
                                    <Badge variant="secondary">
                                      {qualityLabel(result.quality_score, allScores)}
                                    </Badge>
                                  </TableCell>
                                  <TableCell className="text-right stat-value text-sm">
                                    {formatScore(result.quality_score)}
                                  </TableCell>
                                </TableRow>
                              ))}
                            </TableBody>
                          </Table>
                        </ScrollArea>
                      ) : (
                        <Empty className="border bg-muted/5 animate-fade-in-up">
                          <EmptyHeader>
                            <EmptyMedia variant="icon">
                              <ImageIcon aria-hidden="true" className="text-primary/60" />
                            </EmptyMedia>
                            <EmptyTitle>暂无评分结果</EmptyTitle>
                            <EmptyDescription>
                              任务还在处理中，或没有可展示的图片。
                            </EmptyDescription>
                          </EmptyHeader>
                        </Empty>
                      )}
                    </div>
                  ) : null}
                  {selectedJob.kind === "folder" && dashboard.selectedResults.length ? (
                    <Disclosure title="展开任务分析" defaultOpen>
                      <div className="flex flex-col gap-4">
                        <div className="grid gap-4 xl:grid-cols-2">
                          <ScoreDistributionChart
                            data={selectedDistribution}
                            title="质量分布"
                            description="按好 / 中 / 差统计当前任务。"
                          />
                          <RankedScoreChart
                            data={selectedRankedScores}
                            title="排序曲线"
                            description="按得分排序查看质量衰减趋势。"
                          />
                        </div>
                      </div>
                    </Disclosure>
                  ) : null}
                </motion.div>
              ) : (
                <DetailEmpty />
              )}
            </AnimatePresence>
          </CardContent>
        </Card>
      </div>
      <Dialog open={trendOpen} onOpenChange={setTrendOpen}>
        <DialogContent className="sm:max-w-4xl">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <ChartNoAxesCombined aria-hidden="true" className="size-5 text-primary" />
              任务管理 · 总体趋势
            </DialogTitle>
          </DialogHeader>

          <div className="grid gap-3 md:grid-cols-3">
            <StatCard label="总任务" value={String(dashboard.jobs.length)} />
            <StatCard label="已完成任务" value={String(completedJobsWithScores.length)} />
            <StatCard label="整体均分" value={formatScore(overallAverageScore)} />
          </div>

          <JobTrendChart
            data={trendData}
            title="历史任务趋势"
            description="最近任务的平均分变化。"
          />

          <DialogFooter>
            <Button variant="outline" onClick={() => setTrendOpen(false)}>
              关闭
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog open={tagDialogOpen} onOpenChange={setTagDialogOpen}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>批量添加标签</DialogTitle>
            <DialogDescription>为选中的 {selectedJobIds.size} 个任务添加标签。</DialogDescription>
          </DialogHeader>
          <div className="py-2">
            <Input
              placeholder="输入标签，多个用逗号分隔"
              value={batchTagValue}
              onChange={(e) => setBatchTagValue(e.target.value)}
              className="h-10"
            />
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setTagDialogOpen(false)}>
              取消
            </Button>
            <Button
              onClick={() => {
                const tags = batchTagValue.split(",").map((t) => t.trim()).filter(Boolean)
                const promises = Array.from(selectedJobIds).map((id) => updateJobTags(id, tags))
                void Promise.all(promises).then(() => {
                  setTagDialogOpen(false)
                  setBatchTagValue("")
                  setSelectedJobIds(new Set())
                  void dashboard.refreshJobs()
                })
              }}
            >
              确认添加
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
      <ImagePreview image={previewImage} onOpenChange={(open) => !open && setPreviewImage(null)} />
      <Dialog open={Boolean(deleteTarget)} onOpenChange={(open) => !open && setDeleteTargetId(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>
              {deleteTarget?.status === "running"
                ? "强制删除这条运行中任务？"
                : "删除这条历史任务？"}
            </DialogTitle>
            <DialogDescription>
              {deleteTarget?.status === "running"
                ? "系统会先停止任务以释放计算资源，再清理数据库记录和对应上传文件，无法恢复。"
                : "删除后会同时清理数据库记录和对应上传文件，无法恢复。"}
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setDeleteTargetId(null)}>
              取消
            </Button>
            <Button
              variant="destructive"
              onClick={() => {
                if (!deleteTarget) return
                void dashboard.removeJob(deleteTarget.id)
                setDeleteTargetId(null)
              }}
              disabled={!deleteTarget || dashboard.deletingJobId === deleteTarget.id}
            >
              {deleteTarget && dashboard.deletingJobId === deleteTarget.id ? (
                <>
                  <Spinner data-icon="inline-start" />
                  {deleteTarget.status === "running" ? "强制删除中…" : "删除中…"}
                </>
              ) : (
                deleteTarget?.status === "running" ? "确认强制删除" : "确认删除"
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </motion.div>
  )
}
