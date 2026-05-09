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
        "rounded-xl border bg-muted/10 p-3.5",
        className
      )}
    >
      <div className="flex items-center justify-between gap-3">
        <div className="text-sm text-muted-foreground">{label}</div>
        {icon ? <div className="text-muted-foreground">{icon}</div> : null}
      </div>
      <div className="mt-2 break-words text-xl font-semibold tracking-tight tabular-nums">{value}</div>
      {hint ? <div className="mt-2 text-sm text-muted-foreground">{hint}</div> : null}
    </div>
  )
}
