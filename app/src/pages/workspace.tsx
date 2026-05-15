import { useEffect, useMemo, useRef, useState } from "react"
import type { ChangeEvent, DragEvent } from "react"
import { Link } from "react-router-dom"
import { AnimatePresence, motion } from "motion/react"
import { ChevronDown, Download, FolderOpen, ImagePlus, ListFilter, SlidersHorizontal, UploadCloud } from "lucide-react"

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
} from "@/lib/format"
import { cn, downloadCsv } from "@/lib/utils"
import { Disclosure } from "@/components/disclosure"
import { ImagePreview, type PreviewImage } from "@/components/image-preview"
import { StatCard } from "@/components/stat-card"
import { RankedScoreChart, ScoreDistributionChart } from "@/components/charts"
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

/* ─── Status badge infrared coloring ──────────────────────── */
function statusBadgeClass(status: string): string {
  switch (status) {
    case "completed":
      return "bg-chart-2/10 text-chart-2 border-chart-2/30"
    case "running":
      return "bg-primary/10 text-primary border-primary/40 animate-infrared-pulse"
    case "failed":
      return "bg-destructive/10 text-destructive border-destructive/40"
    case "interrupted":
      return "bg-chart-4/10 text-chart-4 border-chart-4/30"
    default:
      return ""
  }
}

/* ─── Section title accent bar (shared) ──────────────────── */
const sectionAccent = "border-l-[3px] border-primary pl-3"

/* ============================================================
   UploadDropzone — infrared glow + scan overlay + drag pulse
   ============================================================ */
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
        "relative cursor-pointer overflow-hidden rounded-xl border-2 border-dashed p-4 transition-all duration-300 ease-out",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
        dashboard.isDragging
          ? "scale-[1.01] border-primary bg-primary/[0.06] shadow-[0_0_30px_var(--primary)/0.15]"
          : "border-border hover:border-primary/60 hover:bg-muted/20 hover:infrared-glow",
        dashboard.isDragging && "animate-infrared-pulse",
        dashboard.isSubmitting && "pointer-events-none opacity-70"
      )}
    >
      {/* Subtle scan overlay */}
      <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(180deg,var(--primary)/0.03_0%,transparent_50%,var(--primary)/0.03_100%)] rounded-xl" />

      <div className="relative flex flex-col gap-4">
        <div className="flex items-start gap-3">
          <div className="flex size-10 shrink-0 items-center justify-center rounded-xl bg-primary/10 text-primary ring-1 ring-primary/20">
            <UploadCloud aria-hidden="true" className="size-5" />
          </div>
          <div className="min-w-0">
            <div className="font-semibold text-foreground">拖拽或点击上传图片</div>
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
              <div className="mt-2 font-data text-sm text-muted-foreground">
                {compactFileSize(dashboard.singleFile.size)}
              </div>
            </div>
          </div>
        ) : (
          <div className="rounded-xl border bg-background p-4">
            <div className="flex items-center justify-between gap-3">
              <div className="text-sm font-medium">
                <span className="font-data tabular-nums">{fileCount}</span> 张已选择
              </div>
              <Badge variant="default">批量</Badge>
            </div>
            {folderPreview.length ? (
              <div className="mt-3 flex flex-col gap-2 text-sm text-muted-foreground">
                {folderPreview.map((item) => (
                  <div key={item.relativePath} className="truncate font-data text-xs">
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

/* ============================================================
   PreviewStrip
   ============================================================ */
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
  const allScores = results.map((r) => r.quality_score)

  if (!previewItems.length) {
    return null
  }

  return (
    <div className="flex flex-col gap-2">
      <div className="custom-scrollbar flex gap-3 overflow-x-auto pb-2">
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
            className="w-52 shrink-0 overflow-hidden rounded-xl border bg-card text-left transition-all duration-200 hover:infrared-glow hover:scale-[1.01] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
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
                <div className="stat-value text-sm">{formatScore(result.quality_score)}</div>
                <Badge variant="secondary">{qualityLabel(result.quality_score, allScores)}</Badge>
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

/* ============================================================
   ResultPanelSummary — infrared status badges + progress
   ============================================================ */
function ResultPanelSummary({ dashboard }: { dashboard: Dashboard }) {
  const selectedJob = dashboard.selectedJob

  if (!selectedJob) {
    return null
  }

  return (
    <div className="mb-5 rounded-xl border bg-muted/15 p-4 transition-shadow hover:infrared-glow">
      <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
        <div className="min-w-0">
          <div className="font-semibold">
            {selectedJob.status === "running"
              ? selectedJob.stage
              : `${statusText(selectedJob.status)} · ${selectedJob.result_count} 条结果`}
          </div>
          <div className="mt-1 truncate text-sm text-muted-foreground">
            Run {selectedJob.run_name}
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <Badge
            variant="outline"
            className={cn(statusBadgeClass(selectedJob.status), "font-medium")}
          >
            {statusText(selectedJob.status)}
          </Badge>
          <Badge variant="outline">{selectedJob.kind === "image" ? "单图" : "文件夹"}</Badge>
          <Badge variant="outline">{backendText(selectedJob.backend)}</Badge>
          <Badge variant="outline" className="tabular-nums font-data">
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

/* ============================================================
   ProcessingState
   ============================================================ */
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

/* ============================================================
   AdvancedOptions
   ============================================================ */
function AdvancedOptions({
  dashboard,
  rustBackend,
}: {
  dashboard: Dashboard
  rustBackend?: BackendHealth | null
}) {
  const [open, setOpen] = useState(false)

  return (
    <div className="rounded-xl border bg-muted/10 transition-shadow hover:infrared-glow">
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
                    if (!value) return
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

/* ============================================================
   EmptyResult
   ============================================================ */
function EmptyResult() {
  return (
    <Empty className="border bg-muted/10">
      <EmptyHeader>
        <EmptyMedia variant="icon">
          <ImagePlus aria-hidden="true" className="text-primary/60" />
        </EmptyMedia>
        <EmptyTitle>等待评分结果</EmptyTitle>
        <EmptyDescription>上传后右侧会显示质量分数、预览与批量结果。</EmptyDescription>
      </EmptyHeader>
    </Empty>
  )
}

/* ============================================================
   RecentJobs
   ============================================================ */
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
            "flex flex-col gap-3 rounded-xl border p-4 text-left transition-all duration-200",
            "hover:infrared-glow hover:scale-[1.01] hover:bg-muted/20",
            "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
            dashboard.selectedJob?.id === job.id && "border-primary bg-primary/5 shadow-[0_0_16px_var(--primary)/0.08]"
          )}
        >
          <div className="flex items-center justify-between gap-2">
            <div className="flex flex-wrap items-center gap-2">
              <Badge variant="outline">{job.kind === "image" ? "单图" : "文件夹"}</Badge>
              <Badge variant="outline" className={statusBadgeClass(job.status)}>
                {statusText(job.status)}
              </Badge>
            </div>
            <div className="stat-value text-sm">{formatScore(job.average_score)}</div>
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

/* ============================================================
   WorkspacePage — main orchestrator
   ============================================================ */
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
        {/* ═══ LEFT PANEL — Upload & Score ═══ */}
        <div className="relative animate-fade-in-up" style={{ animationDelay: "0ms" }}>
          <div className="absolute inset-y-0 left-0 z-10 w-[3px] rounded-full bg-primary/60" />
          <Card className="h-fit xl:sticky xl:top-5 xl:self-start hover:infrared-glow transition-shadow duration-300">
            <CardHeader>
              <CardTitle className={sectionAccent}>上传与评分</CardTitle>
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
                <Button
                  size="lg"
                  onClick={() => void dashboard.submit()}
                  disabled={dashboard.isSubmitting}
                >
                  {dashboard.isSubmitting ? (
                    <>
                      <Spinner data-icon="inline-start" />
                      创建任务中…
                    </>
                  ) : (
                    "开始评分"
                  )}
                </Button>
                <Button
                  variant="outline"
                  size="lg"
                  onClick={() => folderInputRef.current?.click()}
                  disabled={dashboard.isSubmitting}
                >
                  <FolderOpen aria-hidden="true" data-icon="inline-start" />
                  文件夹
                </Button>
              </div>

              <Button variant="ghost" size="sm" onClick={resetUploads} disabled={dashboard.isSubmitting}>
                清空
              </Button>

              <AdvancedOptions dashboard={dashboard} rustBackend={rustBackend} />
            </CardContent>
          </Card>
        </div>

        {/* ═══ RIGHT PANEL — Results ═══ */}
        <div className="flex flex-col gap-5 animate-fade-in-up" style={{ animationDelay: "100ms" }}>
          {/* Results card */}
          <div className="relative">
            <div className="absolute inset-y-0 left-0 z-10 w-[3px] rounded-full bg-primary/60" />
            <Card className="hover:infrared-glow transition-shadow duration-300">
              <CardHeader>
                <CardTitle className={sectionAccent}>结果面板</CardTitle>
                {selectedJob && dashboard.selectedResults.length > 0 ? (
                  <CardAction>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => {
                        const headers = ["文件名", "相对路径", "质量分数", "评分等级"]
                        const rows = dashboard.selectedResults.map((r) => [
                          r.relative_path.split("/").pop() || r.relative_path,
                          r.relative_path,
                          r.quality_score.toFixed(2),
                          qualityLabel(r.quality_score, dashboard.selectedResults.map((sr) => sr.quality_score)),
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
                  </CardAction>
                ) : null}
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
                        className="overflow-hidden rounded-xl border bg-card transition-shadow hover:infrared-glow focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                      >
                        <img
                          src={topResult.public_url}
                          alt={topResult.relative_path}
                          width={360}
                          height={360}
                          className="h-80 w-full object-cover transition hover:scale-[1.02]"
                        />
                      </button>
                      <div className="flex flex-col gap-4">
                        <div>
                          <div className="text-sm text-muted-foreground">质量分数</div>
                          <div className="mt-2 flex items-center gap-3">
                            <div className="stat-value text-4xl">{formatScore(topResult.quality_score)}</div>
                            <Badge variant="secondary">
                              {qualityLabel(topResult.quality_score, dashboard.selectedResults.map((r) => r.quality_score))}
                            </Badge>
                          </div>
                        </div>

                        <Progress value={clampPercentage(topResult.quality_score)} className="h-2" />

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
                      {/* Stat cards with staggered animation */}
                      <div className="grid gap-3 md:grid-cols-3">
                        <div className="animate-fade-in-up" style={{ animationDelay: "0ms" }}>
                          <StatCard
                            label="平均分"
                            value={formatScore(selectedJob.average_score)}
                            hint="整批图像平均质量"
                          />
                        </div>
                        <div className="animate-fade-in-up" style={{ animationDelay: "80ms" }}>
                          <StatCard
                            label="最高分"
                            value={formatScore(selectedJob.best_score)}
                            hint="当前最佳样本"
                          />
                        </div>
                        <div className="animate-fade-in-up" style={{ animationDelay: "160ms" }}>
                          <StatCard
                            label="结果数量"
                            value={String(dashboard.selectedResults.length)}
                            hint={selectedJob.completed_at ? formatTime(selectedJob.completed_at) : "等待完成"}
                          />
                        </div>
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
          </div>

          {/* ═══ Charts section ═══ */}
          {selectedJob?.kind === "folder" && dashboard.selectedResults.length ? (
            <div className="animate-fade-in-up" style={{ animationDelay: "200ms" }}>
              <Disclosure
                title="展开质量分析"
                description="仅展示当前任务的分布与排序分析。"
              >
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
                      description="按得分从高到低查看质量衰减趋势。"
                    />
                  </div>
                </div>
              </Disclosure>
            </div>
          ) : null}

          {/* ═══ Recent jobs section ═══ */}
          <div className="animate-fade-in-up" style={{ animationDelay: "300ms" }}>
            <Disclosure title="展开最近任务">
              <RecentJobs dashboard={dashboard} />
            </Disclosure>
          </div>
        </div>
      </div>

      <ImagePreview image={previewImage} onOpenChange={(open) => !open && setPreviewImage(null)} />
    </motion.div>
  )
}
