import { useEffect, useMemo, useRef, useState } from "react"
import type { ChangeEvent, DragEvent } from "react"
import { Link } from "react-router-dom"
import { AnimatePresence, motion } from "motion/react"
import { ChevronDown, FolderOpen, ImagePlus, ListFilter, SlidersHorizontal, UploadCloud } from "lucide-react"

import type { Dashboard } from "@/hooks/use-dashboard"
import type { BackendHealth } from "@/lib/types"
import { buildRankedScoreData, buildScoreDistribution } from "@/lib/analytics"
import {
  backendLabel,
  backendText,
  clampPercentage,
  compactFileSize,
  formatScore,
  formatTime,
  qualityLabel,
  statusText,
  statusVariant,
} from "@/lib/format"
import { cn } from "@/lib/utils"
import { Disclosure } from "@/components/disclosure"
import { ImagePreview, type PreviewImage } from "@/components/image-preview"
import { StatCard } from "@/components/stat-card"
import { RankedScoreChart, ScoreDistributionChart } from "@/components/charts"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
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
  onPreview,
}: {
  dashboard: Dashboard
  singlePreviewUrl: string | null
  onChoose: () => void
  onPreview: (image: PreviewImage) => void
}) {
  const hasFiles = dashboard.singleFile !== null || dashboard.folderItems.length > 0
  const fileCount = dashboard.singleFile ? 1 : dashboard.folderItems.length
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
        "rounded-xl border border-dashed p-4 transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
        dashboard.isDragging ? "border-primary bg-primary/5" : "border-border bg-muted/10 hover:bg-muted/20",
        dashboard.isSubmitting && "pointer-events-none opacity-70"
      )}
    >
      <div className="flex flex-col gap-4">
        <div className="flex items-start gap-3">
          <div className="flex size-10 items-center justify-center rounded-xl bg-muted text-foreground">
            <UploadCloud aria-hidden="true" className="size-4" />
          </div>
          <div className="min-w-0">
            <div className="font-medium">拖拽或点击上传图片</div>
            <div className="mt-1 text-sm text-muted-foreground">
              单击选文件 · 拖拽自动识别 · 下方按钮选文件夹
            </div>
          </div>
        </div>

        {!hasFiles ? (
          <div className="rounded-xl bg-background px-4 py-3 text-sm text-muted-foreground">
            上传后即可开始评分。
          </div>
        ) : dashboard.singleFile ? (
          <div className="grid gap-3 sm:grid-cols-[120px_minmax(0,1fr)]">
            <button
              type="button"
              onClick={(event) => {
                event.stopPropagation()
                if (!singlePreviewUrl) {
                  return
                }
                onPreview({
                  src: singlePreviewUrl,
                  alt: dashboard.singleFile?.name ?? "预览图片",
                  caption: dashboard.singleFile?.name,
                })
              }}
              className="overflow-hidden rounded-xl border bg-card focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            >
              <img
                src={singlePreviewUrl ?? ""}
                alt={dashboard.singleFile?.name ?? ""}
                width={120}
                height={120}
                className="h-30 w-full object-cover transition hover:scale-[1.02]"
              />
            </button>
            <div className="min-w-0 rounded-xl border bg-background p-4">
              <div className="truncate text-sm font-medium">{dashboard.singleFile.name}</div>
              <div className="mt-2 text-sm text-muted-foreground">{compactFileSize(dashboard.singleFile.size)}</div>
            </div>
          </div>
        ) : (
          <div className="rounded-xl border bg-background p-4">
            <div className="flex items-center justify-between gap-3">
              <div className="text-sm font-medium">{fileCount} 张已选择</div>
              <Badge variant="outline">批量</Badge>
            </div>
            {folderPreview.length ? (
              <div className="mt-3 flex flex-col gap-2 text-sm text-muted-foreground">
                {folderPreview.map((item) => (
                  <div key={item.relativePath} className="truncate">
                    {item.relativePath}
                  </div>
                ))}
                {dashboard.folderItems.length > folderPreview.length ? (
                  <div className="text-xs">还有 {dashboard.folderItems.length - folderPreview.length} 张未展开。</div>
                ) : null}
              </div>
            ) : (
              <div className="mt-3 text-sm text-muted-foreground">推荐直接拖入整理好的 ROI 文件夹。</div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

function PreviewStrip({
  results,
  limit = 6,
  onPreview,
}: {
  results: Dashboard["selectedResults"]
  limit?: number
  onPreview: (image: PreviewImage) => void
}) {
  const previewItems = results.slice(0, limit)

  if (!previewItems.length) {
    return null
  }

  return (
    <div className="flex flex-col gap-2">
      <div className="flex gap-3 overflow-x-auto pb-2">
        {previewItems.map((result) => (
          <button
            key={result.id}
            type="button"
            onClick={() =>
              onPreview({
                src: result.public_url,
                alt: result.relative_path,
                caption: `${result.relative_path} · ${formatScore(result.quality_score)}`,
              })
            }
            className="w-52 shrink-0 overflow-hidden rounded-xl border bg-card text-left transition hover:bg-muted/10 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
          >
            <img
              src={result.public_url}
              alt={result.relative_path}
              width={240}
              height={160}
              loading="lazy"
              className="h-32 w-full object-cover transition hover:scale-[1.02]"
            />
            <div className="p-3">
              <div className="flex items-center justify-between gap-2">
                <div className="text-sm font-medium tabular-nums">{formatScore(result.quality_score)}</div>
                <Badge variant="secondary">{qualityLabel(result.quality_score)}</Badge>
              </div>
              <div className="mt-2 truncate text-sm text-muted-foreground">{result.relative_path}</div>
            </div>
          </button>
        ))}
      </div>
      {results.length > previewItems.length ? (
        <div className="text-xs text-muted-foreground">当前仅显示前 {previewItems.length} 张预览，可横向滚动查看。</div>
      ) : null}
    </div>
  )
}

function ResultPanelSummary({ dashboard }: { dashboard: Dashboard }) {
  const selectedJob = dashboard.selectedJob

  if (!selectedJob) {
    return null
  }

  return (
    <div className="mb-5 rounded-xl border bg-muted/10 p-4">
      <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
        <div className="min-w-0">
          <div className="font-medium">
            {selectedJob.status === "running"
              ? selectedJob.stage
              : `${statusText(selectedJob.status)} · ${selectedJob.result_count} 条结果`}
          </div>
          <div className="mt-1 truncate text-sm text-muted-foreground">
            Run {selectedJob.run_name}
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <Badge variant={statusVariant(selectedJob.status)}>{statusText(selectedJob.status)}</Badge>
          <Badge variant="outline">{selectedJob.kind === "image" ? "单图" : "文件夹"}</Badge>
          <Badge variant="outline">{backendText(selectedJob.backend)}</Badge>
          <Badge variant="outline" className="tabular-nums">
            {selectedJob.processed_count}/{selectedJob.input_count}
          </Badge>
        </div>
      </div>

      {selectedJob.status === "running" ? (
        <Progress value={clampPercentage(selectedJob.progress)} className="mt-4 h-2" />
      ) : null}
    </div>
  )
}

function ProcessingState({
  dashboard,
  onPreview,
}: {
  dashboard: Dashboard
  onPreview: (image: PreviewImage) => void
}) {
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
      className="flex flex-col gap-4"
    >
      {selectedJob.kind === "folder" && dashboard.selectedResults.length ? (
        <PreviewStrip results={dashboard.selectedResults} limit={4} onPreview={onPreview} />
      ) : (
        <div className="rounded-xl border bg-muted/10 p-4 text-sm text-muted-foreground">
          任务进行中，结果会在这里持续更新。
        </div>
      )}
    </motion.div>
  )
}

function AdvancedOptions({
  dashboard,
  rustBackend,
}: {
  dashboard: Dashboard
  rustBackend?: BackendHealth | null
}) {
  const [open, setOpen] = useState(false)

  return (
    <div className="rounded-xl border bg-muted/10">
      <button
        type="button"
        onClick={() => setOpen((current) => !current)}
        className="flex w-full items-center justify-between gap-4 px-4 py-4 text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
      >
        <div className="flex items-start gap-3">
          <div className="flex size-9 items-center justify-center rounded-xl bg-background text-muted-foreground">
            <SlidersHorizontal className="size-4" aria-hidden="true" />
          </div>
          <div className="min-w-0">
            <div className="text-sm font-medium">展开运行设置</div>
            <div className="mt-1 truncate text-sm text-muted-foreground">
              {backendLabel(dashboard.backend, dashboard.device)}
            </div>
          </div>
        </div>
        <ChevronDown
          aria-hidden="true"
          className={cn("size-4 shrink-0 text-muted-foreground transition-transform", open && "rotate-180")}
        />
      </button>

      {open ? (
        <div className="border-t px-4 pb-4 pt-4">
          <FieldGroup>
            <Field>
              <FieldLabel htmlFor="run-name">推理 Run</FieldLabel>
              <FieldContent>
                <Input
                  id="run-name"
                  name="run-name"
                  autoComplete="off"
                  value={dashboard.runName}
                  onChange={(event) => dashboard.setRunName(event.target.value)}
                  placeholder={dashboard.health?.defaultRunName ?? "自动选择最新可用 run…"}
                />
                <FieldDescription>留空时使用当前默认 checkpoint。</FieldDescription>
              </FieldContent>
            </Field>

            <Field>
              <FieldLabel>推理引擎</FieldLabel>
              <FieldContent>
                <ToggleGroup
                  type="single"
                  variant="outline"
                  value={`${dashboard.backend}-${dashboard.device}`}
                  onValueChange={(value) => {
                    const [b, d] = value.split("-")
                    if (b === "python" || b === "rust") {
                      dashboard.setBackend(b)
                      dashboard.setDevice(d as "cpu" | "cuda")
                    }
                  }}
                >
                  <ToggleGroupItem value="python-cuda" aria-label="使用 Python 后端">
                    Python
                  </ToggleGroupItem>
                  <ToggleGroupItem value="rust-cpu" aria-label="使用 Rust CPU 推理">
                    Rust CPU
                  </ToggleGroupItem>
                  <ToggleGroupItem
                    value="rust-cuda"
                    aria-label="使用 Rust CUDA GPU 推理"
                    disabled={!rustBackend?.available || rustBackend?.device !== "cuda"}
                  >
                    Rust CUDA
                  </ToggleGroupItem>
                </ToggleGroup>
                <FieldDescription>
                  {dashboard.backend === "python"
                    ? "PyTorch · CUDA 加速"
                    : dashboard.device === "cuda"
                      ? !rustBackend?.available || rustBackend?.device !== "cuda"
                        ? "CUDA 二进制未构建"
                        : "candle-onnx · GPU 推理"
                      : "candle-onnx · CPU 推理"}
                </FieldDescription>
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
    <Empty className="border bg-muted/10">
      <EmptyHeader>
        <EmptyMedia variant="icon">
          <ImagePlus aria-hidden="true" />
        </EmptyMedia>
        <EmptyTitle>等待评分结果</EmptyTitle>
        <EmptyDescription>上传后右侧会显示质量分数、预览与批量结果。</EmptyDescription>
      </EmptyHeader>
    </Empty>
  )
}

function RecentJobs({ dashboard }: { dashboard: Dashboard }) {
  const recentJobs = dashboard.jobs.slice(0, 4)

  if (!recentJobs.length) {
    return (
      <Empty className="border bg-muted/10">
        <EmptyHeader>
          <EmptyTitle>还没有任务</EmptyTitle>
          <EmptyDescription>提交后会在这里显示。</EmptyDescription>
        </EmptyHeader>
      </Empty>
    )
  }

  return (
    <div className="grid gap-3 sm:grid-cols-2">
      {recentJobs.map((job) => (
        <button
          key={job.id}
          type="button"
          onClick={() => void dashboard.selectJob(job.id)}
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
        </button>
      ))}
    </div>
  )
}

export function WorkspacePage({ dashboard }: { dashboard: Dashboard }) {
  const fileInputRef = useRef<HTMLInputElement>(null)
  const folderInputRef = useRef<HTMLInputElement>(null)
  const [previewImage, setPreviewImage] = useState<PreviewImage | null>(null)

  useEffect(() => {
    folderInputRef.current?.setAttribute("webkitdirectory", "")
    folderInputRef.current?.setAttribute("directory", "")
  }, [])

  useEffect(() => {
    if (!dashboard.singleFile && !dashboard.folderItems.length && fileInputRef.current) {
      fileInputRef.current.value = ""
    }
  }, [dashboard.singleFile, dashboard.folderItems.length])

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

  const selectedJob = dashboard.selectedJob
  const topResult = dashboard.topResult
  const rustBackend = dashboard.health?.backends.rust
  const selectedDistribution = useMemo(
    () => buildScoreDistribution(dashboard.selectedResults),
    [dashboard.selectedResults]
  )
  const selectedRankedScores = useMemo(
    () => buildRankedScoreData(dashboard.selectedResults),
    [dashboard.selectedResults]
  )

  function resetUploads() {
    dashboard.resetUploads()
    if (fileInputRef.current) {
      fileInputRef.current.value = ""
    }
  }

  function handleFileInputChange(event: ChangeEvent<HTMLInputElement>) {
    const files = Array.from(event.target.files ?? [])
    if (files.length === 0) return
    if (files.length === 1) {
      dashboard.selectImageFile(files[0])
      dashboard.selectFolderFiles([])
    } else {
      dashboard.selectImageFile(null)
      dashboard.selectFolderFiles(files)
    }
  }

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

      <div className="grid items-start gap-6 xl:grid-cols-[360px_minmax(0,1fr)]">
        <Card className="h-fit xl:sticky xl:top-5 xl:self-start">
          <CardHeader>
            <CardTitle>上传与评分</CardTitle>
          </CardHeader>
          <CardContent className="flex flex-col gap-5">
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              multiple
              className="hidden"
              onChange={handleFileInputChange}
            />
            <input
              ref={folderInputRef}
              type="file"
              multiple
              className="hidden"
              onChange={handleFileInputChange}
            />

            <UploadDropzone
              dashboard={dashboard}
              singlePreviewUrl={singlePreviewUrl}
              onPreview={setPreviewImage}
              onChoose={() => fileInputRef.current?.click()}
            />

            <div className="grid grid-cols-2 gap-3">
              <Button size="lg" onClick={() => void dashboard.submit()} disabled={dashboard.isSubmitting}>
                {dashboard.isSubmitting ? <><Spinner data-icon="inline-start" />创建任务中…</> : "开始评分"}
              </Button>
              <Button variant="outline" size="lg" onClick={() => folderInputRef.current?.click()} disabled={dashboard.isSubmitting}>
                <FolderOpen aria-hidden="true" data-icon="inline-start" />
                文件夹
              </Button>
            </div>

            <Button variant="ghost" size="sm" onClick={resetUploads} disabled={dashboard.isSubmitting}>
              清空
            </Button>

            <AdvancedOptions
              dashboard={dashboard}
              rustBackend={rustBackend}
            />
          </CardContent>
        </Card>

        <div className="flex flex-col gap-5">
          <Card>
            <CardHeader>
              <CardTitle>结果面板</CardTitle>
            </CardHeader>
            <CardContent>
              <ResultPanelSummary dashboard={dashboard} />
              <AnimatePresence mode="wait">
                {selectedJob?.status === "running" ? (
                  <ProcessingState key={selectedJob.id} dashboard={dashboard} onPreview={setPreviewImage} />
                ) : selectedJob?.kind === "image" && topResult ? (
                  <motion.div
                    key={`image-${topResult.id}`}
                    initial={{ opacity: 0, y: 12 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -12 }}
                    className="grid gap-6 lg:grid-cols-[minmax(0,320px)_minmax(0,1fr)]"
                  >
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
                        className="h-80 w-full object-cover transition hover:scale-[1.01]"
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

                      <Progress value={clampPercentage(topResult.quality_score * 100)} className="h-2" />

                      <div className="grid gap-3 sm:grid-cols-2">
                        <StatCard label="完成时间" value={formatTime(selectedJob.completed_at)} />
                        <StatCard label="文件名" value={topResult.relative_path} />
                      </div>
                    </div>
                  </motion.div>
                ) : selectedJob?.kind === "folder" && dashboard.selectedResults.length ? (
                  <motion.div
                    key={`folder-${selectedJob.id}`}
                    initial={{ opacity: 0, y: 12 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -12 }}
                    className="flex flex-col gap-5"
                  >
                    <div className="grid gap-3 md:grid-cols-3">
                      <StatCard label="平均分" value={formatScore(selectedJob.average_score)} hint="整批图像平均质量" />
                      <StatCard label="最高分" value={formatScore(selectedJob.best_score)} hint="当前最佳样本" />
                      <StatCard
                        label="结果数量"
                        value={String(dashboard.selectedResults.length)}
                        hint={selectedJob.completed_at ? formatTime(selectedJob.completed_at) : "等待完成"}
                      />
                    </div>

                    <PreviewStrip results={dashboard.selectedResults} onPreview={setPreviewImage} />

                    <div>
                      <Button asChild variant="outline">
                        <Link to="/jobs">
                          <ListFilter aria-hidden="true" data-icon="inline-start" />
                          查看完整结果
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

          {selectedJob?.kind === "folder" && dashboard.selectedResults.length ? (
            <Disclosure
              title="展开质量分析"
              description="仅展示当前任务的分布与排序分析。"
            >
              <div className="flex flex-col gap-4">
                <div className="grid gap-4 xl:grid-cols-2">
                  <ScoreDistributionChart
                    data={selectedDistribution}
                    title="质量分布"
                    description="按优秀 / 良好 / 一般 / 偏低统计当前任务。"
                  />
                  <RankedScoreChart
                    data={selectedRankedScores}
                    title="排序曲线"
                    description="按得分从高到低查看质量衰减趋势。"
                  />
                </div>
              </div>
            </Disclosure>
          ) : null}

          <Disclosure
            title="展开最近任务"
          >
            <RecentJobs dashboard={dashboard} />
          </Disclosure>
        </div>
      </div>

      <ImagePreview image={previewImage} onOpenChange={(open) => !open && setPreviewImage(null)} />
    </motion.div>
  )
}
