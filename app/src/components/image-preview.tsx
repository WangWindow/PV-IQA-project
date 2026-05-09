import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"

export type PreviewImage = {
  src: string
  alt: string
  caption?: string
}

export function ImagePreview({
  image,
  onOpenChange,
}: {
  image: PreviewImage | null
  onOpenChange: (open: boolean) => void
}) {
  return (
    <Dialog open={Boolean(image)} onOpenChange={onOpenChange}>
      <DialogContent className="overflow-hidden border-border/80 bg-background/96 p-3 shadow-xl backdrop-blur-xl sm:max-w-5xl">
        <DialogHeader className="px-1 pt-1">
          <DialogTitle className="truncate text-sm font-medium">{image?.alt ?? "图片预览"}</DialogTitle>
          {image?.caption ? <DialogDescription className="truncate">{image.caption}</DialogDescription> : null}
        </DialogHeader>
        {image ? (
          <div className="overflow-hidden rounded-xl border bg-muted/20">
            <img
              src={image.src}
              alt={image.alt}
              className="max-h-[78vh] w-full object-contain"
            />
          </div>
        ) : null}
      </DialogContent>
    </Dialog>
  )
}
