import { useMemo, useState } from "react"
import { AnimatePresence, motion } from "motion/react"
import { Eye, FolderClock, Image as ImageIcon, Trash2 } from "lucide-react"

import type { DemoDashboard } from "@/hooks/use-demo-dashboard"
import {
  averageScore,
  backendText,
  clampPercentage,
  formatTime,
  qualityLabel,
  statusText,
  statusVariant,
} from "@/lib/demo-format"
import { cn } from "@/lib/utils"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Card,
  CardAction,
  CardContent,
  CardDescription,
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
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet"
import { Spinner } from "@/components/ui/spinner"
import { ScrollArea } from "@/components/ui/scroll-area"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"

type HistoryFilter = "all" | "running" | "completed" | "failed"

function JobList({
  dashboard,
  jobs,
  onPick,
}: {
  dashboard: DemoDashboard
  jobs: DemoDashboard["jobs"]
  onPick?: () => void
}) {
  return (
    <div className="flex flex-col gap-3">
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
              "flex flex-col gap-3 rounded-2xl border p-4 text-left transition hover:bg-muted/50",
              dashboard.selectedJob?.id === job.id && "border-primary bg-primary/5"
            )}
          >
            <div className="flex items-center justify-between gap-2">
              <div className="flex items-center gap-2">
                <Badge variant="outline">{job.kind === "image" ? "单图" : "文件夹"}</Badge>
                <Badge variant={statusVariant(job.status)}>{statusText(job.status)}</Badge>
                <Badge variant="outline">{backendText(job.backend)}</Badge>
              </div>
              <div className="text-xs text-muted-foreground">{job.result_count} 条</div>
            </div>
            <div className="flex flex-col gap-1">
              <div className="truncate text-sm font-medium">{job.stage}</div>
              <div className="text-xs text-muted-foreground">{formatTime(job.created_at)}</div>
            </div>
            <div className="flex items-center justify-between gap-3 text-xs text-muted-foreground">
              <span>{job.run_name}</span>
              <span>
                {job.processed_count}/{job.input_count}
              </span>
            </div>
          </button>
        ))}
      </div>
    </div>
  )
}

function DetailEmpty() {
  return (
    <Empty className="border bg-muted/20">
        <EmptyHeader>
          <EmptyMedia variant="icon">
            <FolderClock />
          </EmptyMedia>
          <EmptyTitle>选择一条历史任务</EmptyTitle>
          <EmptyDescription>从列表中选择任务查看详情。</EmptyDescription>
        </EmptyHeader>
      </Empty>
    )
}

export function HistoryPage({ dashboard }: { dashboard: DemoDashboard }) {
  const [filter, setFilter] = useState<HistoryFilter>("all")
  const [deleteTargetId, setDeleteTargetId] = useState<string | null>(null)
  const [mobileSheetOpen, setMobileSheetOpen] = useState(false)

  const filteredJobs = useMemo(() => {
    if (filter === "all") {
      return dashboard.jobs
    }
    return dashboard.jobs.filter((job) => job.status === filter)
  }, [dashboard.jobs, filter])

  const selectedJob = dashboard.selectedJob
  const selectedAverage = averageScore(dashboard.selectedResults)
  const deleteTarget = dashboard.jobs.find((job) => job.id === deleteTargetId) ?? null

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

        <Card>
          <CardHeader>
            <CardTitle>任务管理</CardTitle>
            <CardDescription>筛选、查看和删除历史任务。</CardDescription>
            <CardAction className="hidden md:block">
            <div className="flex items-center gap-2">
              <Badge variant="outline">{dashboard.jobs.length} 总任务</Badge>
              <Badge variant="outline">{dashboard.activeRunningCount} 处理中</Badge>
            </div>
          </CardAction>
        </CardHeader>
        <CardContent className="flex flex-col gap-4">
          <div className="flex items-center justify-between gap-3 md:hidden">
            <div className="flex items-center gap-2">
              <Badge variant="outline">{dashboard.jobs.length} 总任务</Badge>
              <Badge variant="outline">{dashboard.activeRunningCount} 处理中</Badge>
            </div>
            <Sheet open={mobileSheetOpen} onOpenChange={setMobileSheetOpen}>
              <SheetTrigger asChild>
                <Button variant="outline" size="sm">
                  <Eye data-icon="inline-start" />
                  任务列表
                </Button>
              </SheetTrigger>
              <SheetContent side="right">
                <SheetHeader>
                  <SheetTitle>历史任务</SheetTitle>
                  <SheetDescription>选择任务查看详情。</SheetDescription>
                </SheetHeader>
                <div className="px-4">
                  <JobList dashboard={dashboard} jobs={filteredJobs} onPick={() => setMobileSheetOpen(false)} />
                </div>
              </SheetContent>
            </Sheet>
          </div>

          <ToggleGroup
            type="single"
            variant="outline"
            value={filter}
            onValueChange={(value) => {
              if (value === "all" || value === "running" || value === "completed" || value === "failed") {
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
            <ToggleGroupItem value="completed" aria-label="显示已完成任务">
              已完成
            </ToggleGroupItem>
            <ToggleGroupItem value="failed" aria-label="显示失败任务">
              失败
            </ToggleGroupItem>
          </ToggleGroup>
        </CardContent>
      </Card>

      <div className="grid gap-6 lg:grid-cols-[320px_minmax(0,1fr)]">
        <Card className="hidden lg:flex">
          <CardHeader>
            <CardTitle>历史列表</CardTitle>
            <CardDescription>按时间倒序显示。</CardDescription>
          </CardHeader>
          <CardContent>
            {filteredJobs.length ? (
              <ScrollArea className="h-[680px] pr-2">
                <JobList dashboard={dashboard} jobs={filteredJobs} />
              </ScrollArea>
            ) : (
              <Empty className="border bg-muted/20">
                <EmptyHeader>
                  <EmptyTitle>没有匹配任务</EmptyTitle>
                  <EmptyDescription>切换筛选条件或先提交任务。</EmptyDescription>
                </EmptyHeader>
              </Empty>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>任务详情</CardTitle>
            <CardDescription>{selectedJob ? `Run ${selectedJob.run_name}` : "选择一条任务查看详情。"}</CardDescription>
            {selectedJob ? (
              <CardAction>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setDeleteTargetId(selectedJob.id)}
                  disabled={selectedJob.status === "running"}
                >
                  <Trash2 data-icon="inline-start" />
                  删除
                </Button>
              </CardAction>
            ) : null}
          </CardHeader>
          <CardContent>
            <AnimatePresence mode="wait">
              {!selectedJob ? (
                <motion.div key="detail-empty" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
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
                    <Badge variant="outline">{selectedJob.kind === "image" ? "单图" : "文件夹"}</Badge>
                    <Badge variant={statusVariant(selectedJob.status)}>{statusText(selectedJob.status)}</Badge>
                    <Badge variant="outline">{backendText(selectedJob.backend)}</Badge>
                    <Badge variant="outline">{selectedJob.result_count} 结果</Badge>
                  </div>

                  <div className="grid gap-3 md:grid-cols-4">
                    <div className="rounded-2xl border bg-muted/30 p-4">
                      <div className="text-sm text-muted-foreground">创建时间</div>
                      <div className="mt-2 font-medium">{formatTime(selectedJob.created_at)}</div>
                    </div>
                    <div className="rounded-2xl border bg-muted/30 p-4">
                      <div className="text-sm text-muted-foreground">Run</div>
                      <div className="mt-2 break-all font-medium">{selectedJob.run_name}</div>
                    </div>
                    <div className="rounded-2xl border bg-muted/30 p-4">
                      <div className="text-sm text-muted-foreground">处理状态</div>
                      <div className="mt-2 font-medium">{selectedJob.stage}</div>
                    </div>
                    <div className="rounded-2xl border bg-muted/30 p-4">
                      <div className="text-sm text-muted-foreground">处理数量</div>
                      <div className="mt-2 font-medium">
                        {selectedJob.processed_count}/{selectedJob.input_count}
                      </div>
                    </div>
                  </div>

                  {selectedJob.status === "running" ? (
                    <div className="rounded-3xl border bg-muted/20 p-6">
                      <div className="flex items-center gap-3">
                        <Spinner />
                        <div className="flex flex-col gap-1">
                          <div className="font-medium">处理中</div>
                          <div className="text-sm text-muted-foreground">
                            {selectedJob.kind === "folder"
                              ? `已完成 ${selectedJob.processed_count}/${selectedJob.input_count}`
                              : "正在执行模型评分"}
                          </div>
                        </div>
                      </div>
                    </div>
                  ) : null}

                  {selectedJob.error ? (
                    <Alert variant="destructive">
                      <AlertTitle>任务失败</AlertTitle>
                      <AlertDescription>{selectedJob.error}</AlertDescription>
                    </Alert>
                  ) : null}

                  {selectedJob.kind === "image" && dashboard.topResult ? (
                    <div className="grid gap-6 lg:grid-cols-[minmax(0,280px)_minmax(0,1fr)]">
                      <img
                        src={dashboard.topResult.public_url}
                        alt={dashboard.topResult.relative_path}
                        className="h-72 w-full rounded-3xl border object-cover"
                      />
                      <div className="flex flex-col gap-4">
                        <div className="flex items-center gap-3">
                          <div className="text-4xl font-semibold">
                            {dashboard.topResult.quality_score.toFixed(4)}
                          </div>
                          <Badge variant="secondary">
                            {qualityLabel(dashboard.topResult.quality_score)}
                          </Badge>
                        </div>
                        <div className="flex flex-col gap-2">
                          <div className="flex items-center justify-between text-sm text-muted-foreground">
                            <span>质量分位</span>
                            <span>{clampPercentage(dashboard.topResult.quality_score * 100).toFixed(1)}%</span>
                          </div>
                          <div className="h-2 rounded-full bg-muted">
                            <div
                              className="h-2 rounded-full bg-primary"
                              style={{ width: `${clampPercentage(dashboard.topResult.quality_score * 100)}%` }}
                            />
                          </div>
                        </div>
                        <div className="rounded-2xl border bg-muted/30 p-4">
                          <div className="text-sm text-muted-foreground">文件路径</div>
                          <div className="mt-2 break-all font-medium">
                            {dashboard.topResult.relative_path}
                          </div>
                        </div>
                      </div>
                    </div>
                  ) : null}

                  {selectedJob.kind === "folder" ? (
                    <div className="flex flex-col gap-4">
                      <div className="grid gap-3 md:grid-cols-3">
                        <div className="rounded-2xl border bg-muted/30 p-4">
                          <div className="text-sm text-muted-foreground">平均分</div>
                          <div className="mt-2 text-2xl font-semibold">
                            {selectedAverage?.toFixed(4) ?? "—"}
                          </div>
                        </div>
                        <div className="rounded-2xl border bg-muted/30 p-4">
                          <div className="text-sm text-muted-foreground">最高分</div>
                          <div className="mt-2 text-2xl font-semibold">
                            {dashboard.selectedResults[0]?.quality_score.toFixed(4) ?? "—"}
                          </div>
                        </div>
                        <div className="rounded-2xl border bg-muted/30 p-4">
                          <div className="text-sm text-muted-foreground">最低分</div>
                          <div className="mt-2 text-2xl font-semibold">
                            {dashboard.selectedResults.at(-1)?.quality_score.toFixed(4) ?? "—"}
                          </div>
                        </div>
                      </div>

                      {dashboard.selectedResults.length ? (
                        <ScrollArea className="h-[420px]">
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
                                    <img
                                      src={result.public_url}
                                      alt={result.relative_path}
                                      className="h-12 w-16 rounded-xl border object-cover"
                                    />
                                  </TableCell>
                                  <TableCell className="max-w-[320px] truncate">
                                    {result.relative_path}
                                  </TableCell>
                                  <TableCell>
                                    <Badge variant="secondary">{qualityLabel(result.quality_score)}</Badge>
                                  </TableCell>
                                  <TableCell className="text-right font-medium">
                                    {result.quality_score.toFixed(4)}
                                  </TableCell>
                                </TableRow>
                              ))}
                            </TableBody>
                          </Table>
                        </ScrollArea>
                      ) : (
                        <Empty className="border bg-muted/20">
                          <EmptyHeader>
                            <EmptyMedia variant="icon">
                              <ImageIcon />
                            </EmptyMedia>
                            <EmptyTitle>暂无评分结果</EmptyTitle>
                            <EmptyDescription>任务还在处理中，或没有可展示的图片。</EmptyDescription>
                          </EmptyHeader>
                        </Empty>
                      )}
                    </div>
                  ) : null}
                </motion.div>
              )}
            </AnimatePresence>
          </CardContent>
        </Card>
      </div>

      <Dialog open={Boolean(deleteTarget)} onOpenChange={(open) => !open && setDeleteTargetId(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>删除这条历史任务？</DialogTitle>
            <DialogDescription>
              删除后会同时清理数据库记录和对应上传文件，无法恢复。
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
                  删除中
                </>
              ) : (
                "确认删除"
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </motion.div>
  )
}
