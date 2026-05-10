import { useMemo, useState } from "react"
import { AnimatePresence, motion } from "motion/react"
import { ChartNoAxesCombined, Download, Eye, FolderClock, Image as ImageIcon, PauseCircle, Play, RotateCcw, Trash2 } from "lucide-react"

import type { Dashboard } from "@/hooks/use-dashboard"
import type { JobRecord } from "@/lib/types"
import { buildJobTrendData, buildRankedScoreData, buildScoreDistribution } from "@/lib/analytics"
import {
  backendText,
  clampPercentage,
  formatScore,
  formatTime,
  qualityLabel,
  statusText,
  statusVariant,
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

type HistoryFilter = "all" | "running" | "interrupted" | "completed" | "failed"

function HistoryFilterControls({
  filter,
  setFilter,
  className,
}: {
  filter: HistoryFilter
  setFilter: (value: HistoryFilter) => void
  className?: string
}) {
  return (
    <ToggleGroup
      type="single"
      variant="outline"
      value={filter}
      className={className}
      onValueChange={(value) => {
        if (
          value === "all" ||
          value === "running" ||
          value === "interrupted" ||
          value === "completed" ||
          value === "failed"
        ) {
          setFilter(value)
        }
      }}
    >
      <ToggleGroupItem value="all" aria-label="显示全部任务">
        全部
      </ToggleGroupItem>
      <ToggleGroupItem value="running" aria-label="显示处理中任务">
        处理中
      </ToggleGroupItem>
      <ToggleGroupItem value="interrupted" aria-label="显示中断任务">
        中断
      </ToggleGroupItem>
      <ToggleGroupItem value="completed" aria-label="显示已完成任务">
        已完成
      </ToggleGroupItem>
      <ToggleGroupItem value="failed" aria-label="显示失败任务">
        失败
      </ToggleGroupItem>
    </ToggleGroup>
  )
}

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
    <div className="flex flex-col gap-3">
      {jobs.map((job) => (
        <button
          key={job.id}
          type="button"
          onClick={() => {
            void dashboard.selectJob(job.id)
            onPick?.()
          }}
          className={cn(
            "flex flex-col gap-3 rounded-xl border p-4 text-left transition hover:bg-muted/20 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
            dashboard.selectedJob?.id === job.id && "border-primary bg-primary/5"
          )}
        >
          <div className="flex items-center justify-between gap-2">
            <div className="flex flex-wrap items-center gap-2">
              <Badge variant="outline">{job.kind === "image" ? "单图" : "文件夹"}</Badge>
              <Badge variant={statusVariant(job.status)}>{statusText(job.status)}</Badge>
            </div>
            <div className="text-sm font-medium tabular-nums">{formatScore(job.average_score)}</div>
          </div>
          <div className="min-w-0">
            <div className="truncate text-sm font-medium">{job.stage}</div>
            <div className="mt-1 text-xs text-muted-foreground">{formatTime(job.created_at)}</div>
          </div>
          {job.status === "running" ? (
            <Progress value={clampPercentage(job.progress)} className="h-2" />
          ) : (
            <div className="text-xs text-muted-foreground tabular-nums">{job.result_count} 条结果</div>
          )}
        </button>
      ))}
    </div>
  )
}

function DetailEmpty() {
  return (
    <Empty className="border bg-muted/10">
      <EmptyHeader>
        <EmptyMedia variant="icon">
          <FolderClock aria-hidden="true" />
        </EmptyMedia>
        <EmptyTitle>选择一条任务</EmptyTitle>
        <EmptyDescription>从左侧列表中选择任务后，这里显示详细结果与可选分析。</EmptyDescription>
      </EmptyHeader>
    </Empty>
  )
}

function JobMetaLine({ job }: { job: JobRecord }) {
  return (
    <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
      <span>Run {job.run_name}</span>
      <span>创建 {formatTime(job.created_at)}</span>
      <span>更新 {formatTime(job.updated_at)}</span>
      {job.completed_at ? <span>完成 {formatTime(job.completed_at)}</span> : null}
      <span>
        已处理 {job.processed_count}/{job.input_count}
      </span>
    </div>
  )
}

export function HistoryPage({ dashboard }: { dashboard: Dashboard }) {
  const [filter, setFilter] = useState<HistoryFilter>("all")
  const [deleteTargetId, setDeleteTargetId] = useState<string | null>(null)
  const [mobileSheetOpen, setMobileSheetOpen] = useState(false)
  const [trendOpen, setTrendOpen] = useState(false)
  const [previewImage, setPreviewImage] = useState<PreviewImage | null>(null)

  const filteredJobs = useMemo(() => {
    if (filter === "all") {
      return dashboard.jobs
    }
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
    if (!completedJobsWithScores.length) {
      return null
    }
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

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -16 }}
      transition={{ duration: 0.22, ease: "easeOut" }}
      className="flex flex-col gap-6"
    >
      {dashboard.error ? (
        <Alert variant="destructive">
          <AlertTitle>操作失败</AlertTitle>
          <AlertDescription>{dashboard.error}</AlertDescription>
        </Alert>
      ) : null}

      <div className="grid items-start gap-6 lg:grid-cols-[320px_minmax(0,1fr)]">
        <Card className="hidden lg:flex lg:sticky lg:top-5 lg:self-start">
          <CardHeader>
            <CardTitle>任务列表</CardTitle>
            <CardAction className="flex flex-wrap items-center gap-2">
              <Badge variant="outline">{dashboard.jobs.length} 总任务</Badge>
              {dashboard.activeRunningCount ? <Badge variant="outline">{dashboard.activeRunningCount} 处理中</Badge> : null}
              <Button variant="ghost" size="sm" onClick={() => setTrendOpen(true)}>
                <ChartNoAxesCombined aria-hidden="true" data-icon="inline-start" />
                总体趋势
              </Button>
            </CardAction>
          </CardHeader>
          <CardContent className="flex flex-col gap-4">
            <HistoryFilterControls filter={filter} setFilter={setFilter} className="flex-wrap" />
            {filteredJobs.length ? (
              <ScrollArea className="h-[calc(100vh-14rem)] pr-2">
                <JobList dashboard={dashboard} jobs={filteredJobs} />
              </ScrollArea>
            ) : (
              <Empty className="border bg-muted/10">
                <EmptyHeader>
                  <EmptyTitle>没有匹配任务</EmptyTitle>
                  <EmptyDescription>切换筛选条件或先创建任务。</EmptyDescription>
                </EmptyHeader>
              </Empty>
            )}
          </CardContent>
        </Card>

        <Card>
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
                  <SheetContent side="right">
                    <SheetHeader>
                      <SheetTitle>历史任务</SheetTitle>
                    </SheetHeader>
                    <div className="flex flex-col gap-4 px-4">
                      <div className="flex flex-wrap items-center gap-2">
                        <Badge variant="outline">{dashboard.jobs.length} 总任务</Badge>
                        {dashboard.activeRunningCount ? (
                          <Badge variant="outline">{dashboard.activeRunningCount} 处理中</Badge>
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
              {selectedJob ? (
                <>
                  {selectedJob.status === "running" ? (
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => void dashboard.stopJob(selectedJob.id)}
                      disabled={dashboard.mutatingJobId === selectedJob.id || dashboard.deletingJobId === selectedJob.id}
                    >
                      <PauseCircle aria-hidden="true" data-icon="inline-start" />
                      停止
                    </Button>
                  ) : null}
                  {selectedJob.status === "interrupted" ? (
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => void dashboard.resumeJob(selectedJob.id)}
                      disabled={dashboard.mutatingJobId === selectedJob.id || dashboard.deletingJobId === selectedJob.id}
                    >
                      <Play aria-hidden="true" data-icon="inline-start" />
                      继续处理
                    </Button>
                  ) : null}
                  {selectedJob.status !== "running" ? (
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => void dashboard.rerunJob(selectedJob.id)}
                      disabled={dashboard.mutatingJobId === selectedJob.id || dashboard.deletingJobId === selectedJob.id}
                    >
                      <RotateCcw aria-hidden="true" data-icon="inline-start" />
                      重新运行
                    </Button>
                  ) : null}
                </>
              ) : null}
              {selectedJob ? (
                <Button
                  variant={selectedJob.status === "running" ? "destructive" : "ghost"}
                  size="sm"
                  onClick={() => setDeleteTargetId(selectedJob.id)}
                  disabled={dashboard.mutatingJobId === selectedJob.id}
                >
                  <Trash2 aria-hidden="true" data-icon="inline-start" />
                  {selectedJob.status === "running" ? "强制删除" : "删除"}
                </Button>
              ) : null}
              {selectedJob && dashboard.selectedResults.length > 0 ? (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => {
                    const headers = ["文件名", "相对路径", "质量分数", "评分等级"]
                    const rows = dashboard.selectedResults.map((r) => [
                      r.relative_path.split("/").pop() || r.relative_path,
                      r.relative_path,
                      r.quality_score.toFixed(2),
                      qualityLabel(r.quality_score),
                    ])
                    downloadCsv(
                      `pv-iqa-${selectedJob.id.slice(0, 8)}.csv`,
                      headers,
                      rows
                    )
                  }}
                >
                  <Download aria-hidden="true" data-icon="inline-start" />
                  导出 CSV
                </Button>
              ) : null}
            </CardAction>
          </CardHeader>
          <CardContent>
            <AnimatePresence mode="wait">
              {!selectedJob ? (
                <motion.div key="empty" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                  <DetailEmpty />
                </motion.div>
              ) : (
                <motion.div
                  key={selectedJob.id}
                  initial={{ opacity: 0, y: 12 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -12 }}
                  className="flex flex-col gap-5"
                >
                  <div className="flex flex-wrap items-center gap-2">
                    <Badge variant={statusVariant(selectedJob.status)}>{statusText(selectedJob.status)}</Badge>
                    <Badge variant="outline">{selectedJob.kind === "image" ? "单图" : "文件夹"}</Badge>
                    <Badge variant="outline">{backendText(selectedJob.backend)}</Badge>
                  </div>
                  <JobMetaLine job={selectedJob} />

                  {selectedJob.status === "running" ? (
                    <div className="rounded-xl border bg-muted/10 p-4">
                      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                        <div className="flex items-center gap-3">
                          <Spinner />
                          <div>
                            <div className="font-medium">{selectedJob.stage}</div>
                            <div className="text-sm text-muted-foreground">进度会持续刷新。</div>
                          </div>
                        </div>
                        <div className="text-sm text-muted-foreground tabular-nums">
                          {Math.round(selectedJob.progress)}%
                        </div>
                      </div>
                      <Progress value={clampPercentage(selectedJob.progress)} className="mt-4 h-2" />
                    </div>
                  ) : null}

                  {selectedJob.error ? (
                    selectedJob.status === "failed" ? (
                      <Alert variant="destructive">
                        <AlertTitle>任务失败</AlertTitle>
                        <AlertDescription>{selectedJob.error}</AlertDescription>
                      </Alert>
                    ) : (
                      <Alert>
                        <AlertTitle>任务已中断</AlertTitle>
                        <AlertDescription>{selectedJob.error}</AlertDescription>
                      </Alert>
                    )
                  ) : null}

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
                        className="overflow-hidden rounded-xl border focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                      >
                        <img
                          src={topResult.public_url}
                          alt={topResult.relative_path}
                          width={360}
                          height={360}
                          className="h-72 w-full object-cover transition hover:scale-[1.01]"
                        />
                      </button>
                      <div className="flex flex-col gap-4">
                        <div>
                          <div className="text-sm text-muted-foreground">质量分数</div>
                          <div className="mt-2 flex items-center gap-3">
                            <div className="text-4xl font-semibold tabular-nums">{formatScore(topResult.quality_score)}</div>
                            <Badge variant="secondary">{qualityLabel(topResult.quality_score)}</Badge>
                          </div>
                        </div>
                        <Progress value={clampPercentage(topResult.quality_score)} className="h-2" />
                        <div className="rounded-xl border bg-muted/10 p-4">
                          <div className="text-sm text-muted-foreground">文件路径</div>
                          <div className="mt-2 break-all font-medium">{topResult.relative_path}</div>
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
                        <ScrollArea className="h-[420px] rounded-xl border">
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
                                  <TableRow key={result.id}>
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
                                        className="overflow-hidden rounded-lg border focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                                      >
                                        <img
                                          src={result.public_url}
                                          alt={result.relative_path}
                                          width={64}
                                          height={48}
                                          loading="lazy"
                                          className="h-12 w-16 object-cover transition hover:scale-[1.03]"
                                        />
                                      </button>
                                    </TableCell>
                                  <TableCell className="max-w-[320px] truncate">{result.relative_path}</TableCell>
                                  <TableCell>
                                    <Badge variant="secondary">{qualityLabel(result.quality_score)}</Badge>
                                  </TableCell>
                                  <TableCell className="text-right font-medium tabular-nums">
                                    {formatScore(result.quality_score)}
                                  </TableCell>
                                </TableRow>
                              ))}
                            </TableBody>
                          </Table>
                        </ScrollArea>
                      ) : (
                        <Empty className="border bg-muted/10">
                          <EmptyHeader>
                            <EmptyMedia variant="icon">
                              <ImageIcon aria-hidden="true" />
                            </EmptyMedia>
                            <EmptyTitle>暂无评分结果</EmptyTitle>
                            <EmptyDescription>任务还在处理中，或没有可展示的图片。</EmptyDescription>
                          </EmptyHeader>
                        </Empty>
                      )}
                    </div>
                  ) : null}

                  {selectedJob.kind === "folder" && dashboard.selectedResults.length ? (
                    <Disclosure title="展开任务分析">
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
              )}
            </AnimatePresence>
          </CardContent>
        </Card>
      </div>

      <Dialog open={trendOpen} onOpenChange={setTrendOpen}>
        <DialogContent className="sm:max-w-4xl">
          <DialogHeader>
            <DialogTitle>任务管理 · 总体趋势</DialogTitle>
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

      <ImagePreview image={previewImage} onOpenChange={(open) => !open && setPreviewImage(null)} />

      <Dialog open={Boolean(deleteTarget)} onOpenChange={(open) => !open && setDeleteTargetId(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{deleteTarget?.status === "running" ? "强制删除这条运行中任务？" : "删除这条历史任务？"}</DialogTitle>
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
                if (!deleteTarget) {
                  return
                }
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
