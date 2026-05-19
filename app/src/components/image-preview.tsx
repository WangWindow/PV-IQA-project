import { useCallback, useEffect, useRef, useState } from "react"
import {
  BarChart3,
  FileImage,
  Info,
  Loader2,
  RotateCcw,
  ZoomIn,
  ZoomOut,
} from "lucide-react"

import type { ImageMetadata } from "@/lib/types"
import { fetchImageMetadata } from "@/lib/api"
import { cn } from "@/lib/utils"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogTitle,
} from "@/components/ui/dialog"

export type PreviewImage = {
  src: string
  alt: string
  caption?: string
}

function brightnessLabel(value: number | null): string {
  if (value === null) return "—"
  if (value > 180) return "偏亮"
  if (value < 70) return "偏暗"
  return "适中"
}

function contrastLabel(value: number | null): string {
  if (value === null) return "—"
  if (value > 80) return "高对比"
  if (value < 30) return "低对比"
  return "适中"
}

function snrLabel(value: number | null): string {
  if (value === null) return "—"
  if (value > 5) return "高信噪比"
  if (value > 2) return "中等"
  return "低信噪比"
}

type MetricStatus = "good" | "warn" | "bad"

function brightnessStatus(value: number | null): MetricStatus {
  if (value === null) return "good"
  if (value > 180 || value < 70) return "bad"
  if (value > 160 || value < 90) return "warn"
  return "good"
}

function contrastStatus(value: number | null): MetricStatus {
  if (value === null) return "good"
  if (value > 80 || value < 30) return "bad"
  if (value > 65 || value < 40) return "warn"
  return "good"
}

function snrStatus(value: number | null): MetricStatus {
  if (value === null) return "good"
  if (value > 5) return "good"
  if (value > 2) return "warn"
  return "bad"
}

const statusColor: Record<MetricStatus, string> = {
  good: "bg-emerald-500",
  warn: "bg-amber-500",
  bad: "bg-rose-500",
}

function MetricBar({
  label,
  value,
  status,
  detail,
}: {
  label: string
  value: string
  status: MetricStatus
  detail: string
}) {
  const normalized = status === "bad" ? 100 : status === "warn" ? 50 : 30
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-xs">
        <span className="text-muted-foreground">{label}</span>
        <span className="font-data text-xs text-foreground">{value}</span>
      </div>
      <div className="h-1.5 w-full overflow-hidden rounded-full bg-muted">
        <div
          className={cn("h-full rounded-full transition-all duration-500", statusColor[status])}
          style={{ width: `${normalized}%` }}
        />
      </div>
      <div className="text-[11px] text-muted-foreground">{detail}</div>
    </div>
  )
}

function MiniHistogram({ data, color, label }: { data: number[]; color: string; label: string }) {
  if (!data || data.length === 0) return null
  const bins = Math.min(data.length, 32)
  const binSize = Math.ceil(data.length / bins)
  const sampled = Array.from({ length: bins }, (_, i) => {
    const start = i * binSize
    const slice = data.slice(start, start + binSize)
    return slice.reduce((a, b) => a + b, 0) / slice.length
  })
  const max = Math.max(...sampled, 1)

  return (
    <div className="space-y-1">
      <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
        <span className="inline-block size-2 rounded-full" style={{ backgroundColor: color }} />
        <span>{label}</span>
      </div>
      <div className="flex h-8 items-end gap-px">
        {sampled.map((value, index) => (
          <div
            key={index}
            className="flex-1 rounded-t transition-all"
            style={{
              backgroundColor: color,
              height: `${(value / max) * 100}%`,
              opacity: 0.5 + (value / max) * 0.5,
            }}
          />
        ))}
      </div>
    </div>
  )
}

function ZoomableImage({ src, alt }: { src: string; alt: string }) {
  const [scale, setScale] = useState(1)
  const [offset, setOffset] = useState({ x: 0, y: 0 })
  const [dragging, setDragging] = useState(false)
  const dragStart = useRef({ x: 0, y: 0, offsetX: 0, offsetY: 0 })
  const containerRef = useRef<HTMLDivElement>(null)

  const reset = useCallback(() => {
    setScale(1)
    setOffset({ x: 0, y: 0 })
  }, [])

  const zoomIn = useCallback(() => setScale((s) => Math.min(s * 1.25, 8)), [])
  const zoomOut = useCallback(() => setScale((s) => Math.max(s / 1.25, 0.5)), [])

  useEffect(() => {
    function onWheel(e: WheelEvent) {
      if (!containerRef.current?.contains(e.target as Node)) return
      e.preventDefault()
      const delta = e.deltaY > 0 ? 0.9 : 1.1
      setScale((s) => Math.min(Math.max(s * delta, 0.5), 8))
    }
    window.addEventListener("wheel", onWheel, { passive: false })
    return () => window.removeEventListener("wheel", onWheel)
  }, [])

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === "=" || e.key === "+") zoomIn()
      if (e.key === "-" || e.key === "_") zoomOut()
      if (e.key === "0") reset()
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [zoomIn, zoomOut, reset])

  function onMouseDown(e: React.MouseEvent) {
    if (scale <= 1) return
    setDragging(true)
    dragStart.current = {
      x: e.clientX,
      y: e.clientY,
      offsetX: offset.x,
      offsetY: offset.y,
    }
  }

  function onMouseMove(e: React.MouseEvent) {
    if (!dragging) return
    setOffset({
      x: dragStart.current.offsetX + (e.clientX - dragStart.current.x),
      y: dragStart.current.offsetY + (e.clientY - dragStart.current.y),
    })
  }

  function onMouseUp() {
    setDragging(false)
  }

  return (
    <div
      ref={containerRef}
      className="relative flex-1 min-h-0 flex items-center justify-center overflow-hidden bg-muted/20 select-none"
      onMouseDown={onMouseDown}
      onMouseMove={onMouseMove}
      onMouseUp={onMouseUp}
      onMouseLeave={onMouseUp}
    >
      <img
        src={src}
        alt={alt}
        draggable={false}
        className="block max-h-full max-w-full object-contain transition-transform duration-75"
        style={{
          transform: `translate(${offset.x}px, ${offset.y}px) scale(${scale})`,
          cursor: scale > 1 ? (dragging ? "grabbing" : "grab") : "zoom-in",
        }}
      />

      <div className="absolute bottom-4 left-1/2 -translate-x-1/2 flex items-center gap-1 rounded-lg border border-border/60 bg-background/90 p-1 shadow-lg backdrop-blur">
        <button
          type="button"
          onClick={zoomOut}
          className="inline-flex size-8 items-center justify-center rounded-md text-muted-foreground hover:bg-muted hover:text-foreground"
          aria-label="缩小"
        >
          <ZoomOut className="size-4" />
        </button>
        <span className="min-w-[3ch] px-1 text-center text-xs font-medium tabular-nums">
          {Math.round(scale * 100)}%
        </span>
        <button
          type="button"
          onClick={zoomIn}
          className="inline-flex size-8 items-center justify-center rounded-md text-muted-foreground hover:bg-muted hover:text-foreground"
          aria-label="放大"
        >
          <ZoomIn className="size-4" />
        </button>
        <div className="mx-1 h-4 w-px bg-border" />
        <button
          type="button"
          onClick={reset}
          className="inline-flex size-8 items-center justify-center rounded-md text-muted-foreground hover:bg-muted hover:text-foreground"
          aria-label="重置缩放"
        >
          <RotateCcw className="size-4" />
        </button>
      </div>
    </div>
  )
}

export function ImagePreview({
  image,
  onOpenChange,
}: {
  image: PreviewImage | null
  onOpenChange: (open: boolean) => void
}) {
  const [metadata, setMetadata] = useState<ImageMetadata | null>(null)
  const [loadingMeta, setLoadingMeta] = useState(false)
  const [showMeta, setShowMeta] = useState(false)

  const loadMetadata = useCallback(async () => {
    if (!image?.src) return
    setLoadingMeta(true)
    try {
      const meta = await fetchImageMetadata(image.src)
      setMetadata(meta)
    } catch {
      setMetadata(null)
    } finally {
      setLoadingMeta(false)
    }
  }, [image?.src])

  useEffect(() => {
    if (image && showMeta) {
      void loadMetadata()
    }
  }, [image, showMeta, loadMetadata])

  useEffect(() => {
    if (!image) {
      setShowMeta(false)
      setMetadata(null)
    }
  }, [image])

  return (
    <Dialog open={Boolean(image)} onOpenChange={onOpenChange}>
      <DialogContent
        className="flex flex-col gap-0 w-[96vw] h-[92vh] max-w-none sm:max-w-5xl overflow-hidden border-border/60 bg-background/96 p-0 shadow-2xl backdrop-blur-xl"
        showCloseButton={false}
      >
        <div className="relative z-20 flex shrink-0 items-center justify-between gap-4 border-b border-border/60 px-5 py-3">
          <div className="min-w-0">
            <DialogTitle className="truncate text-base font-semibold">
              {image?.alt ?? "图片预览"}
            </DialogTitle>
            {image?.caption ? (
              <DialogDescription className="truncate text-sm">
                {image.caption}
              </DialogDescription>
            ) : null}
          </div>
          {image ? (
            <button
              type="button"
              onClick={() => setShowMeta((v) => !v)}
              className={cn(
                "inline-flex shrink-0 items-center gap-1.5 rounded-lg border px-3 py-1.5 text-xs font-medium transition-all duration-200",
                showMeta
                  ? "border-primary bg-primary/10 text-primary"
                  : "border-border bg-card text-muted-foreground hover:border-primary/40 hover:text-primary"
              )}
            >
              {loadingMeta ? (
                <Loader2 className="size-3.5 animate-spin" />
              ) : (
                <Info className="size-3.5" />
              )}
              {showMeta ? "隐藏信息" : "查看信息"}
            </button>
          ) : null}
        </div>

        {image ? (
          <div className="relative z-20 flex flex-1 min-h-0 flex-col lg:flex-row overflow-hidden">
            <div className={cn("flex flex-1 min-h-0 flex-col", showMeta ? "lg:w-[65%]" : "w-full")}>
              <ZoomableImage src={image.src} alt={image.alt} />
            </div>

            {showMeta ? (
              <div className="flex min-h-0 w-full flex-col gap-4 overflow-y-auto border-t lg:border-t-0 lg:border-l border-border/60 bg-card/50 p-4 lg:w-[35%] custom-scrollbar">
                <section className="space-y-2.5">
                  <div className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-widest text-primary">
                    <FileImage className="size-3.5" />
                    基础信息
                  </div>
                  {metadata ? (
                    <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-2 text-xs">
                      <dt className="text-muted-foreground">文件名</dt>
                      <dd className="font-medium truncate font-data">{metadata.filename}</dd>
                      <dt className="text-muted-foreground">格式</dt>
                      <dd className="font-data">{metadata.format ?? "—"}</dd>
                      <dt className="text-muted-foreground">大小</dt>
                      <dd className="font-data">{metadata.size_human}</dd>
                      <dt className="text-muted-foreground">尺寸</dt>
                      <dd className="stat-value text-xs">{metadata.width} × {metadata.height}</dd>
                      {metadata.dpi && (
                        <>
                          <dt className="text-muted-foreground">DPI</dt>
                          <dd className="font-data">{metadata.dpi[0].toFixed(0)} × {metadata.dpi[1].toFixed(0)}</dd>
                        </>
                      )}
                      {metadata.mode && (
                        <>
                          <dt className="text-muted-foreground">色彩</dt>
                          <dd className="font-data">{metadata.mode}</dd>
                        </>
                      )}
                    </dl>
                  ) : loadingMeta ? (
                    <div className="flex items-center gap-2 text-xs text-muted-foreground">
                      <Loader2 className="size-3 animate-spin" />
                      加载中...
                    </div>
                  ) : (
                    <p className="text-xs text-muted-foreground">无法获取元数据</p>
                  )}
                </section>

                {metadata ? (
                  <section className="space-y-2.5">
                    <div className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-widest text-primary">
                      <BarChart3 className="size-3.5" />
                      光学指标
                    </div>
                    <div className="space-y-3">
                      <MetricBar
                        label="亮度"
                        value={metadata.brightness?.toFixed(1) ?? "—"}
                        status={brightnessStatus(metadata.brightness)}
                        detail={brightnessLabel(metadata.brightness)}
                      />
                      <MetricBar
                        label="对比度"
                        value={metadata.contrast?.toFixed(1) ?? "—"}
                        status={contrastStatus(metadata.contrast)}
                        detail={contrastLabel(metadata.contrast)}
                      />
                      <MetricBar
                        label="信噪比"
                        value={metadata.snr_estimate?.toFixed(1) ?? "—"}
                        status={snrStatus(metadata.snr_estimate)}
                        detail={snrLabel(metadata.snr_estimate)}
                      />
                    </div>
                  </section>
                ) : null}

                {metadata?.histogram ? (
                  <section className="space-y-2.5">
                    <div className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-widest text-primary">
                      <BarChart3 className="size-3.5" />
                      颜色直方图
                    </div>
                    <div className="space-y-2.5">
                      <MiniHistogram data={metadata.histogram.r} color="var(--primary)" label="红色通道" />
                      <MiniHistogram data={metadata.histogram.g} color="var(--chart-2)" label="绿色通道" />
                      <MiniHistogram data={metadata.histogram.b} color="var(--chart-4)" label="蓝色通道" />
                      <MiniHistogram data={metadata.histogram.luminance} color="var(--muted-foreground)" label="亮度" />
                    </div>
                  </section>
                ) : null}
              </div>
            ) : null}
          </div>
        ) : null}
      </DialogContent>
    </Dialog>
  )
}
