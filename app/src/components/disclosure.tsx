import { useId, useState, type ReactNode } from "react"
import { ChevronDown } from "lucide-react"

import { cn } from "@/lib/utils"

export function Disclosure({
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
    <section
      className={cn(
        "group relative overflow-hidden rounded-xl border border-border bg-card",
        "transition-all duration-300 ease-out",
        "hover:bg-accent/20",
        className
      )}
    >
      {/* Accent bar on left side when open */}
      <div
        className={cn(
          "absolute inset-y-0 left-0 w-0.5 bg-primary transition-all duration-300 ease-out",
          open ? "opacity-100" : "opacity-0"
        )}
      />

      <button
        type="button"
        aria-expanded={open}
        aria-controls={contentId}
        onClick={() => setOpen((current) => !current)}
        className="flex w-full items-center justify-between gap-4 px-5 py-4 text-left transition-colors duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
      >
        <div className="min-w-0">
          <div className="font-semibold text-foreground">{title}</div>
          {description ? (
            <div className="mt-1 text-sm text-muted-foreground">{description}</div>
          ) : null}
        </div>
        <ChevronDown
          aria-hidden="true"
          className={cn(
            "size-4 shrink-0 text-muted-foreground transition-transform duration-300 ease-out",
            open && "rotate-180"
          )}
        />
      </button>

      {/* Smooth height animation using CSS grid trick */}
      <div
        id={contentId}
        role="region"
        aria-labelledby={contentId}
        className={cn(
          "grid transition-all duration-300 ease-out",
          open ? "grid-rows-[1fr] opacity-100" : "grid-rows-[0fr] opacity-0"
        )}
      >
        <div className="overflow-hidden">
          <div className="border-t border-border px-5 py-4">{children}</div>
        </div>
      </div>
    </section>
  )
}
