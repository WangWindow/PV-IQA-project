import { useEffect, useMemo, useRef, useState } from "react"
import type { ChangeEvent, DragEvent } from "react"
import { Link } from "react-router-dom"
import { AnimatePresence, motion } from "motion/react"
import { ChevronDown, FolderOpen, ImagePlus, ListFilter, SlidersHorizontal, UploadCloud } from "lucide-react"

import type { DemoDashboard } from "@/hooks/use-demo-dashboard"
import type { BackendHealth } from "@/lib/types"
import {
  backendDeviceText,
  backendHintText,
  backendText,
  clampPercentage,
  compactFileSize,
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
  Empty,
  EmptyDescription,
  EmptyHeader,
  EmptyMedia,
  EmptyTitle,
} from "@/components/ui/empty"
import {
  Field,
  FieldContent,
  FieldDescription,
  FieldGroup,
  FieldLabel,
} from "@/components/ui/field"
import { Input } from "@/components/ui/input"
import { Progress } from "@/components/ui/progress"
import { Spinner } from "@/components/ui/spinner"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"

function UploadDropzone({
  dashboard,
  singlePreviewUrl,
  onChoose,
}: {
  dashboard: DemoDashboard
  singlePreviewUrl: string | null
  onChoose: () => void
}) {
  const isImageMode = dashboard.mode === "image"
  const folderPreview = dashboard.folderItems.slice(0, 3)

  function handleDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault()
    dashboard.setDragging(false)
    void dashboard.handleDrop(event.dataTransfer)
  }

  return (
    <div
      role="button"
      tabIndex={0}
      onClick={onChoose}
      onDragOver={(event) => {
        event.preventDefault()
        event.dataTransfer.dropEffect = "copy"
        dashboard.setDragging(true)
      }}
      onDragLeave={() => dashboard.setDragging(false)}
      onDrop={handleDrop}
      onKeyDown={(event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault()
          onChoose()
        }
      }}
      className={cn(
        "rounded-3xl border border-dashed bg-muted/40 p-5 transition",
        dashboard.isDragging
          ? "border-primary bg-primary/5"
          : "border-border hover:border-primary/40 hover:bg-muted/70",
        dashboard.isSubmitting && "pointer-events-none opacity-70"
      )}
    >
      <div className="flex flex-col gap-4">
        <div className="flex items-center gap-3">
          <div className="flex size-12 items-center justify-center rounded-2xl bg-primary/10 text-primary">
            <UploadCloud />
          </div>
          <div className="flex flex-col gap-1">
            <div className="text-base font-medium">
              {isImageMode ? "拖拽一张 ROI 图片" : "拖拽文件夹或多张 ROI 图片"}
            </div>
            <div className="text-sm text-muted-foreground">
              {isImageMode ? "点击后选择图片" : "点击后选择整个文件夹"}
            </div>
          </div>
        </div>

        {isImageMode ? (
          dashboard.singleFile ? (
            <div className="grid gap-4 sm:grid-cols-[128px_minmax(0,1fr)]">
              <div className="overflow-hidden rounded-2xl border bg-card">
                <img
                  src={singlePreviewUrl ?? ""}
                  alt={dashboard.singleFile.name}
                  className="h-32 w-full object-cover"
                />
              </div>
              <div className="flex flex-col gap-2 rounded-2xl border bg-background/80 p-4">
                <div className="truncate text-sm font-medium">{dashboard.singleFile.name}</div>
                <div className="text-sm text-muted-foreground">
                  {compactFileSize(dashboard.singleFile.size)}
                </div>
              </div>
            </div>
          ) : (
            <div className="rounded-2xl bg-background/80 p-4 text-sm text-muted-foreground">
              拖入后即可提交评分。
            </div>
          )
        ) : (
          <div className="rounded-2xl border bg-background/80 p-4">
            <div className="flex items-center justify-between gap-3">
              <div className="text-sm font-medium">已选择 {dashboard.folderItems.length} 张</div>
              <Badge variant="outline">批量评分</Badge>
            </div>
            {folderPreview.length ? (
              <div className="mt-3 flex flex-col gap-1 text-sm text-muted-foreground">
                {folderPreview.map((item) => (
                  <div key={item.relativePath} className="truncate">
                    {item.relativePath}
                  </div>
                ))}
                {dashboard.folderItems.length > folderPreview.length ? (
                  <div className="text-xs">
                    还有 {dashboard.folderItems.length - folderPreview.length} 张图片未展开
                  </div>
                ) : null}
              </div>
            ) : (
              <div className="mt-3 text-sm text-muted-foreground">
                保留相对路径，适合批量评估。
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

function ProcessingState({ dashboard }: { dashboard: DemoDashboard }) {
  const selectedJob = dashboard.selectedJob

  if (!selectedJob || selectedJob.status !== "running") {
    return null
  }

  return (
    <motion.div
      key={`running-${selectedJob.id}`}
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -12 }}
      className="flex flex-col gap-5"
    >
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <Badge variant="outline">
            <Spinner data-icon="inline-start" />
            处理中
          </Badge>
          <Badge variant="outline">{selectedJob.kind === "image" ? "单图" : "文件夹"}</Badge>
          <Badge variant="outline">{backendText(selectedJob.backend)}</Badge>
        </div>
        <div className="text-sm text-muted-foreground">
          {selectedJob.processed_count}/{selectedJob.input_count}
        </div>
      </div>

      <div className="rounded-3xl border bg-muted/30 p-5">
        <div className="flex items-center gap-3">
          <div className="flex size-10 items-center justify-center rounded-2xl bg-background">
            <Spinner />
          </div>
          <div className="flex flex-col gap-1">
            <div className="text-base font-medium">{selectedJob.stage}</div>
            <div className="text-sm text-muted-foreground">
              {selectedJob.kind === "folder"
                ? `已完成 ${selectedJob.processed_count} / ${selectedJob.input_count}`
                : "模型正在处理当前图片"}
            </div>
          </div>
        </div>

        <div className="mt-5 overflow-hidden rounded-full bg-background">
          <motion.div
            className="h-2 rounded-full bg-primary/70"
            animate={{ x: ["-35%", "135%"] }}
            transition={{ duration: 1.4, repeat: Number.POSITIVE_INFINITY, ease: "easeInOut" }}
            style={{ width: "38%" }}
          />
        </div>
      </div>
    </motion.div>
  )
}

function AdvancedOptions({
  dashboard,
  activeBackend,
  pythonBackend,
  rustBackend,
}: {
  dashboard: DemoDashboard
  activeBackend?: BackendHealth | null
  pythonBackend?: BackendHealth | null
  rustBackend?: BackendHealth | null
}) {
  const [open, setOpen] = useState(false)
  const runSummary = dashboard.runName.trim()
    ? `Run ${dashboard.runName.trim()}`
    : `默认 Run ${dashboard.health?.defaultRunName ?? "未发现"}`

  return (
    <div className="rounded-2xl border bg-muted/20">
      <button
        type="button"
        onClick={() => setOpen((current) => !current)}
        className="flex w-full items-center justify-between gap-4 px-4 py-4 text-left"
      >
        <div className="flex items-start gap-3">
          <div className="flex size-9 items-center justify-center rounded-xl bg-background text-muted-foreground">
            <SlidersHorizontal className="size-4" />
          </div>
          <div className="flex flex-col gap-1">
            <div className="text-sm font-medium">高级选项</div>
            <div className="text-sm text-muted-foreground">
              {backendText(dashboard.backend)} · {runSummary}
            </div>
          </div>
        </div>
        <ChevronDown className={cn("size-4 shrink-0 text-muted-foreground transition", open && "rotate-180")} />
      </button>

      {open ? (
        <div className="border-t px-4 pb-4 pt-4">
          <FieldGroup>
            <Field>
              <FieldLabel htmlFor="run-name">推理 Run</FieldLabel>
              <FieldContent>
                <Input
                  id="run-name"
                  value={dashboard.runName}
                  onChange={(event) => dashboard.setRunName(event.target.value)}
                  placeholder={dashboard.health?.defaultRunName ?? "自动选择最新可用 run"}
                />
                <FieldDescription>留空时使用当前默认 checkpoint。</FieldDescription>
              </FieldContent>
            </Field>

            <Field>
              <FieldLabel>推理后端</FieldLabel>
              <FieldContent>
                <ToggleGroup
                  type="single"
                  variant="outline"
                  value={dashboard.backend}
                  onValueChange={(value) => {
                    if (value === "python" || value === "rust") {
                      dashboard.setBackend(value)
                    }
                  }}
                >
                  <ToggleGroupItem value="python" aria-label="使用 Python 后端">
                    Python
                  </ToggleGroupItem>
                  <ToggleGroupItem value="rust" aria-label="使用 Rust 后端">
                    Rust
                  </ToggleGroupItem>
                </ToggleGroup>
                <FieldDescription>{backendHintText(activeBackend)}</FieldDescription>
                <div className="flex flex-wrap gap-2 pt-2">
                  <Badge variant={dashboard.backend === "python" ? "secondary" : "outline"}>
                    Python {backendDeviceText(pythonBackend)}
                  </Badge>
                  <Badge variant={dashboard.backend === "rust" ? "secondary" : "outline"}>
                    Rust {backendDeviceText(rustBackend)}
                  </Badge>
                </div>
              </FieldContent>
            </Field>
          </FieldGroup>
        </div>
      ) : null}
    </div>
  )
}

function EmptyResult() {
  return (
    <Empty className="border bg-muted/20">
      <EmptyHeader>
        <EmptyMedia variant="icon">
          <ImagePlus />
        </EmptyMedia>
        <EmptyTitle>暂无结果</EmptyTitle>
        <EmptyDescription>提交任务后会在这里显示结果。</EmptyDescription>
      </EmptyHeader>
    </Empty>
  )
}

function RecentJobs({ dashboard }: { dashboard: DemoDashboard }) {
  const recentJobs = dashboard.jobs.slice(0, 4)

  return (
    <Card>
      <CardHeader>
        <CardTitle>最近任务</CardTitle>
        <CardDescription>最近 4 条任务。</CardDescription>
        <CardAction>
          <Button asChild variant="ghost" size="sm">
            <Link to="/jobs">
              <ListFilter data-icon="inline-start" />
              查看全部
            </Link>
          </Button>
        </CardAction>
      </CardHeader>
      <CardContent>
        {recentJobs.length ? (
          <div className="grid gap-3 sm:grid-cols-2">
            {recentJobs.map((job) => (
              <button
                key={job.id}
                type="button"
                onClick={() => void dashboard.selectJob(job.id)}
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
                  <div className="text-xs text-muted-foreground">{job.result_count} 结果</div>
                </div>
                <div className="flex flex-col gap-1">
                  <div className="truncate text-sm font-medium">{job.stage}</div>
                  <div className="text-xs text-muted-foreground">{formatTime(job.created_at)}</div>
                </div>
              </button>
            ))}
          </div>
        ) : (
            <Empty className="border bg-muted/20">
              <EmptyHeader>
                <EmptyTitle>还没有任务</EmptyTitle>
                <EmptyDescription>提交后会在这里显示。</EmptyDescription>
              </EmptyHeader>
            </Empty>
          )}
      </CardContent>
    </Card>
  )
}

export function WorkspacePage({ dashboard }: { dashboard: DemoDashboard }) {
  const imageInputRef = useRef<HTMLInputElement>(null)
  const folderInputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    folderInputRef.current?.setAttribute("webkitdirectory", "")
    folderInputRef.current?.setAttribute("directory", "")
  }, [])

  const singlePreviewUrl = useMemo(
    () => (dashboard.singleFile ? URL.createObjectURL(dashboard.singleFile) : null),
    [dashboard.singleFile]
  )

  useEffect(() => {
    return () => {
      if (singlePreviewUrl) {
        URL.revokeObjectURL(singlePreviewUrl)
      }
    }
  }, [singlePreviewUrl])

  function resetUploads() {
    dashboard.resetUploads()
    if (imageInputRef.current) {
      imageInputRef.current.value = ""
    }
    if (folderInputRef.current) {
      folderInputRef.current.value = ""
    }
  }

  function handleImageInputChange(event: ChangeEvent<HTMLInputElement>) {
    dashboard.selectImageFile(event.target.files?.item(0) ?? null)
  }

  function handleFolderInputChange(event: ChangeEvent<HTMLInputElement>) {
    dashboard.selectFolderFiles(Array.from(event.target.files ?? []))
  }

  const selectedJob = dashboard.selectedJob
  const isImageResult = selectedJob?.kind === "image" && dashboard.topResult
  const isFolderResult = selectedJob?.kind === "folder" && dashboard.selectedResults.length
  const pythonBackend = dashboard.health?.backends.python
  const rustBackend = dashboard.health?.backends.rust
  const activeBackend = dashboard.backend === "python" ? pythonBackend : rustBackend

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

        <div className="grid gap-6 lg:grid-cols-[360px_minmax(0,1fr)]">
          <Card className="h-fit">
            <CardHeader>
              <CardTitle>上传与评分</CardTitle>
              <CardDescription>先选择 ROI 图像，再提交评分任务。</CardDescription>
            </CardHeader>
            <CardContent className="flex flex-col gap-6">
              <Field>
                <FieldLabel>评分模式</FieldLabel>
                <FieldContent>
                  <ToggleGroup
                    type="single"
                    variant="outline"
                    value={dashboard.mode}
                    onValueChange={(value) => {
                      if (value === "image" || value === "folder") {
                        dashboard.setMode(value)
                      }
                    }}
                  >
                    <ToggleGroupItem value="image" aria-label="单图评分">
                      <ImagePlus data-icon="inline-start" />
                      单图
                    </ToggleGroupItem>
                    <ToggleGroupItem value="folder" aria-label="文件夹评分">
                      <FolderOpen data-icon="inline-start" />
                      文件夹
                    </ToggleGroupItem>
                  </ToggleGroup>
                </FieldContent>
              </Field>

              <input
                ref={imageInputRef}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={handleImageInputChange}
              />
              <input
                ref={folderInputRef}
                type="file"
                multiple
                className="hidden"
                onChange={handleFolderInputChange}
              />

              <UploadDropzone
                dashboard={dashboard}
                singlePreviewUrl={singlePreviewUrl}
                onChoose={() => {
                  if (dashboard.mode === "image") {
                    imageInputRef.current?.click()
                  } else {
                    folderInputRef.current?.click()
                  }
                }}
              />

              <div className="flex flex-col gap-3 sm:flex-row">
                <Button
                  className="flex-1"
                  size="lg"
                  onClick={() => void dashboard.submit()}
                  disabled={dashboard.isSubmitting}
                >
                  {dashboard.isSubmitting ? (
                    <>
                      <Spinner data-icon="inline-start" />
                      正在创建任务
                    </>
                  ) : (
                    "立即评分"
                  )}
                </Button>
                <Button variant="outline" size="lg" onClick={resetUploads} disabled={dashboard.isSubmitting}>
                  清空
                </Button>
              </div>

              <AdvancedOptions
                dashboard={dashboard}
                activeBackend={activeBackend}
                pythonBackend={pythonBackend}
                rustBackend={rustBackend}
              />
            </CardContent>
          </Card>

          <div className="flex flex-col gap-6">
            <Card>
            <CardHeader>
              <CardTitle>当前结果</CardTitle>
              <CardDescription>显示当前任务或最近一次结果。</CardDescription>
              {selectedJob ? (
                <CardAction>
                  <div className="flex items-center gap-2">
                    <Badge variant={statusVariant(selectedJob.status)}>{statusText(selectedJob.status)}</Badge>
                    <Badge variant="outline">{backendText(selectedJob.backend)}</Badge>
                  </div>
                </CardAction>
              ) : null}
            </CardHeader>
            <CardContent>
              <AnimatePresence mode="wait">
                {selectedJob?.status === "running" ? (
                  <ProcessingState key={selectedJob.id} dashboard={dashboard} />
                ) : isImageResult ? (
                  <motion.div
                    key={`image-${dashboard.topResult?.id}`}
                    initial={{ opacity: 0, y: 12 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -12 }}
                    className="grid gap-6 lg:grid-cols-[minmax(0,360px)_minmax(0,1fr)]"
                  >
                    <img
                      src={dashboard.topResult?.public_url}
                      alt={dashboard.topResult?.relative_path}
                      className="h-88 w-full rounded-3xl border object-cover"
                    />
                    <div className="flex flex-col gap-5">
                      <div className="flex items-center gap-3">
                        <div className="text-5xl font-semibold">
                          {dashboard.topResult?.quality_score.toFixed(4)}
                        </div>
                        <Badge variant="secondary">{qualityLabel(dashboard.topResult?.quality_score ?? 0)}</Badge>
                      </div>
                      <div className="flex flex-col gap-2">
                        <div className="flex items-center justify-between text-sm text-muted-foreground">
                          <span>质量分位</span>
                          <span>{clampPercentage((dashboard.topResult?.quality_score ?? 0) * 100).toFixed(1)}%</span>
                        </div>
                        <Progress value={clampPercentage((dashboard.topResult?.quality_score ?? 0) * 100)} />
                      </div>
                      <div className="grid gap-3 md:grid-cols-2">
                        <div className="rounded-2xl border bg-muted/30 p-4">
                          <div className="text-sm text-muted-foreground">文件名</div>
                          <div className="mt-2 truncate font-medium">
                            {dashboard.topResult?.relative_path}
                          </div>
                        </div>
                        <div className="rounded-2xl border bg-muted/30 p-4">
                          <div className="text-sm text-muted-foreground">完成时间</div>
                          <div className="mt-2 font-medium">{formatTime(selectedJob.completed_at)}</div>
                        </div>
                      </div>
                    </div>
                  </motion.div>
                ) : isFolderResult ? (
                  <motion.div
                    key={`folder-${selectedJob?.id}`}
                    initial={{ opacity: 0, y: 12 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -12 }}
                    className="flex flex-col gap-5"
                  >
                    <div className="grid gap-3 md:grid-cols-3">
                      <div className="rounded-2xl border bg-muted/30 p-4">
                        <div className="text-sm text-muted-foreground">平均分</div>
                        <div className="mt-2 text-2xl font-semibold">
                          {dashboard.summaryScore?.toFixed(4)}
                        </div>
                      </div>
                      <div className="rounded-2xl border bg-muted/30 p-4">
                        <div className="text-sm text-muted-foreground">结果数量</div>
                        <div className="mt-2 text-2xl font-semibold">{dashboard.selectedResults.length}</div>
                      </div>
                      <div className="rounded-2xl border bg-muted/30 p-4">
                        <div className="text-sm text-muted-foreground">完成时间</div>
                        <div className="mt-2 text-lg font-medium">{formatTime(selectedJob?.completed_at ?? null)}</div>
                      </div>
                    </div>
                    <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
                      {dashboard.selectedResults.slice(0, 6).map((result) => (
                        <div key={result.id} className="overflow-hidden rounded-3xl border bg-card">
                          <img
                            src={result.public_url}
                            alt={result.relative_path}
                            className="h-40 w-full object-cover"
                          />
                          <div className="flex flex-col gap-2 p-4">
                            <div className="flex items-center justify-between gap-2">
                              <div className="font-medium">{result.quality_score.toFixed(4)}</div>
                              <Badge variant="secondary">{qualityLabel(result.quality_score)}</Badge>
                            </div>
                            <div className="truncate text-sm text-muted-foreground">
                              {result.relative_path}
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                    <div>
                      <Button asChild variant="outline">
                        <Link to="/jobs">
                          <ListFilter data-icon="inline-start" />
                          去任务管理查看完整结果
                        </Link>
                      </Button>
                    </div>
                  </motion.div>
                ) : (
                  <EmptyResult key="empty-result" />
                )}
              </AnimatePresence>
            </CardContent>
          </Card>

          <RecentJobs dashboard={dashboard} />
        </div>
      </div>
    </motion.div>
  )
}
