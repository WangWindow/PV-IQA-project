import { useMemo, useState } from "react"
import { AnimatePresence, motion } from "motion/react"
import {
  ChartNoAxesCombined,
  Download,
  Eye,
  FolderClock,
  Image as ImageIcon,
  PauseCircle,
  Play,
  RotateCcw,
  Trash2,
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
            "data-[state=on]:bg-[var(--primary)] data-[state=on]:text-[var(--primary-foreground)] data-[state=on]:infrared-glow",
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
}: {
  dashboard: Dashboard
  jobs: Dashboard["jobs"]
  onPick?: () => void
}) {
  return (
    <div className="flex flex-col gap-2.5">
      {jobs.map((job, idx) => {
        const accentColor = STATUS_BORDER[job.status] ?? "border-l-transparent"
        return (
          <button
            key={job.id}
            type="button"
            onClick={() => {
              void dashboard.selectJob(job.id)
              onPick?.()
            }}
            className={cn(
              "group relative flex flex-col gap-2.5 rounded-xl border border-border bg-card p-4 text-left",
              "border-l-[3px]",
              accentColor,
              "transition-all duration-200 ease-out",
              "hover:infrared-glow hover:scale-[1.01]",
              "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
              "animate-fade-in-up",
              dashboard.selectedJob?.id === job.id && "bg-accent/20 border-primary",
            )}
            style={{ animationDelay: `${idx * 50}ms` }}
          >
            {/* header row */}
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

            {/* body row */}
            <div className="min-w-0">
              <div className="truncate text-sm font-semibold">{job.stage}</div>
              <div className="mt-1 font-data text-xs text-muted-foreground">
                {formatTime(job.created_at)}
              </div>
            </div>

            {/* footer row */}
            {job.status === "running" ? (
              <Progress value={clampPercentage(job.progress)} className="h-1.5" />
            ) : (
              <div className="font-data text-xs text-muted-foreground">
                {job.result_count} 条结果
              </div>
            )}
          </button>
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
        infrared && "hover:infrared-glow hover:text-[var(--primary)]"
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
        <div className="font-data text-sm text-[var(--primary)] tabular-nums">
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
  const [deleteTargetId, setDeleteTargetId] = useState<string | null>(null)
  const [mobileSheetOpen, setMobileSheetOpen] = useState(false)
  const [trendOpen, setTrendOpen] = useState(false)
  const [previewImage, setPreviewImage] = useState<PreviewImage | null>(null)

  /* ── derived state ──────────────────────── */

  const filteredJobs = useMemo(() => {
    if (filter === "all") return dashboard.jobs
    return dashboard.jobs.filter((job) => job.status === filter)
  }, [dashboard.jobs, filter])

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

  /* ── render ─────────────────────────────── */

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -16 }}
      transition={{ duration: 0.22, ease: "easeOut" }}
      className="flex flex-col gap-6"
    >
      {/* ── global error ──────────────────── */}
      {dashboard.error ? (
        <Alert variant="destructive" className="animate-fade-in-up">
          <AlertTitle>操作失败</AlertTitle>
          <AlertDescription>{dashboard.error}</AlertDescription>
        </Alert>
      ) : null}

      {/* ── Section header ─────────────────── */}
      <div className="flex items-center gap-3 pt-1">
        <span className="block h-5 w-1 rounded-full bg-[var(--primary)]" />
        <h2 className="text-lg font-bold tracking-tight">历史任务</h2>
        <span className="font-data text-xs text-muted-foreground">
          {dashboard.jobs.length} 任务 · {dashboard.activeRunningCount ?? 0} 处理中
        </span>
      </div>

      {/* ── two-column layout ──────────────── */}
      <div className="grid items-start gap-6 lg:grid-cols-[340px_minmax(0,1fr)]">
        {/* ── Left: job list ───────────────── */}
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
            </CardAction>
          </CardHeader>
          <CardContent className="flex flex-col gap-4">
            <HistoryFilterControls filter={filter} setFilter={setFilter} className="flex-wrap" />
            {filteredJobs.length ? (
              <ScrollArea className="h-[calc(100vh-14rem)] custom-scrollbar pr-1">
                <JobList dashboard={dashboard} jobs={filteredJobs} />
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

        {/* ── Right: detail panel ──────────── */}
        <Card className="animate-fade-in-up" style={{ animationDelay: "100ms" }}>
          <CardHeader>
            <CardTitle>任务详情</CardTitle>
            <CardAction className="flex flex-wrap items-center gap-2">
              {/* mobile sheet trigger */}
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
                      <HistoryFilterControls filter={filter} setFilter={setFilter} className="flex-wrap" />
                      <JobList dashboard={dashboard} jobs={filteredJobs} onPick={() => setMobileSheetOpen(false)} />
                    </div>
                  </SheetContent>
                </Sheet>
              </div>

              {/* action buttons */}
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
                  {/* status badges */}
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

                  {/* running progress */}
                  {selectedJob.status === "running" ? (
                    <RunningWidget job={selectedJob} />
                  ) : null}

                  {/* error / interrupted */}
                  {selectedJob.error ? <ErrorAlert job={selectedJob} /> : null}

                  {/* ── single-image result ──── */}
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
                        className="group relative overflow-hidden rounded-xl border border-border transition-all duration-300 hover:infrared-glow focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
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

                  {/* ── folder results ────────── */}
                  {selectedJob.kind === "folder" ? (
                    <div className="flex flex-col gap-5">
                      <div className="grid gap-3 md:grid-cols-3">
                        <StatCard label="平均分" value={formatScore(selectedJob.average_score)} />
                        <StatCard label="最高分" value={formatScore(selectedJob.best_score)} />
                        <StatCard label="结果数量" value={String(dashboard.selectedResults.length)} />
                      </div>

                      {dashboard.selectedResults.length ? (
                        <ScrollArea className="h-[420px] rounded-xl border border-border custom-scrollbar">
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
                                      className="overflow-hidden rounded-lg border border-border transition-all duration-200 hover:infrared-glow focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
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

                  {/* ── folder analytics ───────── */}
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

      {/* ── Trend dialog ────────────────────── */}
      <Dialog open={trendOpen} onOpenChange={setTrendOpen}>
        <DialogContent className="sm:max-w-4xl">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <ChartNoAxesCombined aria-hidden="true" className="size-5 text-[var(--primary)]" />
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

      {/* ── Image preview ──────────────────── */}
      <ImagePreview image={previewImage} onOpenChange={(open) => !open && setPreviewImage(null)} />

      {/* ── Delete confirmation ────────────── */}
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
