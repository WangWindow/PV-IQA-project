import type { ReactNode } from "react"

import { cn } from "@/lib/utils"

export function StatCard({
  label,
  value,
  hint,
  icon,
  className,
}: {
  label: string
  value: string
  hint?: string
  icon?: ReactNode
  className?: string
}) {
  return (
    <div
      className={cn(
        "group relative rounded-xl border border-border bg-card p-4",
        "transition-all duration-300 ease-out",
        "hover:scale-[1.01] hover:infrared-glow",
        className
      )}
    >
      {/* Top row: label + icon */}
      <div className="flex items-start justify-between gap-3">
        <div className="text-sm text-muted-foreground">{label}</div>
        {icon ? (
          <div className="shrink-0 text-primary/80 transition-colors duration-300 group-hover:text-primary">
            {icon}
          </div>
        ) : null}
      </div>

      {/* Value */}
      <div className="stat-value mt-2 break-words text-xl">{value}</div>

      {/* Hint */}
      {hint ? (
        <div className="mt-2 text-sm text-muted-foreground">{hint}</div>
      ) : null}
    </div>
  )
}
