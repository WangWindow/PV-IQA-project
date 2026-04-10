import { useId, useState, type ReactNode } from "react"
import { ChevronDown } from "lucide-react"

import { cn } from "@/lib/utils"

export function DemoDisclosure({
  title,
  description,
  defaultOpen = false,
  children,
  className,
}: {
  title: string
  description?: string
  defaultOpen?: boolean
  children: ReactNode
  className?: string
}) {
  const [open, setOpen] = useState(defaultOpen)
  const contentId = useId()

  return (
    <section className={cn("overflow-hidden rounded-xl border bg-card", className)}>
      <button
        type="button"
        aria-expanded={open}
        aria-controls={contentId}
        onClick={() => setOpen((current) => !current)}
        className="flex w-full items-center justify-between gap-4 px-4 py-4 text-left transition hover:bg-muted/25 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
      >
        <div className="min-w-0">
          <div className="font-medium">{title}</div>
          {description ? <div className="mt-1 text-sm text-muted-foreground">{description}</div> : null}
        </div>
        <ChevronDown
          aria-hidden="true"
          className={cn("size-4 shrink-0 text-muted-foreground transition-transform", open && "rotate-180")}
        />
      </button>

      {open ? (
        <div id={contentId} className="border-t px-4 py-4">
          {children}
        </div>
      ) : null}
    </section>
  )
}
