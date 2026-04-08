import { useState, type ReactNode } from "react"
import { Link, NavLink } from "react-router-dom"
import { FolderClock, LayoutDashboard, Menu } from "lucide-react"

import type { DemoDashboard } from "@/hooks/use-demo-dashboard"
import { cn } from "@/lib/utils"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet"

const NAV_ITEMS = [
  {
    to: "/workspace",
    label: "评分工作台",
    icon: LayoutDashboard,
  },
  {
    to: "/jobs",
    label: "任务管理",
    icon: FolderClock,
  },
] as const

export function AppShell({
  dashboard,
  children,
}: {
  dashboard: DemoDashboard
  children: ReactNode
}) {
  const [mobileNavOpen, setMobileNavOpen] = useState(false)

  return (
    <div className="min-h-screen bg-muted/30">
      <div className="mx-auto flex min-h-screen max-w-7xl flex-col gap-6 px-4 py-5 lg:px-6">
        <header className="flex items-center justify-between rounded-3xl border bg-card/90 px-4 py-3 shadow-sm">
          <div className="flex items-center gap-3">
            <Link to="/workspace" className="flex flex-col">
              <span className="text-sm text-muted-foreground">PV-IQA</span>
              <span className="text-lg font-semibold tracking-tight">掌静脉质量评分</span>
            </Link>
          </div>

          <div className="hidden items-center gap-2 md:flex">
            {dashboard.activeRunningCount ? (
              <Badge variant="outline">{dashboard.activeRunningCount} 处理中</Badge>
            ) : null}
            {NAV_ITEMS.map((item) => {
              const Icon = item.icon
              return (
                <NavLink key={item.to} to={item.to}>
                  {({ isActive }) => (
                    <Button variant={isActive ? "secondary" : "ghost"} size="sm">
                      <Icon data-icon="inline-start" />
                      {item.label}
                    </Button>
                  )}
                </NavLink>
              )
            })}
          </div>

          <div className="flex items-center gap-2 md:hidden">
            {dashboard.activeRunningCount ? (
              <Badge variant="outline">{dashboard.activeRunningCount} 处理中</Badge>
            ) : null}
            <Sheet open={mobileNavOpen} onOpenChange={setMobileNavOpen}>
              <SheetTrigger asChild>
                <Button variant="ghost" size="icon-sm">
                  <Menu />
                  <span className="sr-only">打开导航菜单</span>
                </Button>
              </SheetTrigger>
              <SheetContent side="right">
                <SheetHeader>
                  <SheetTitle>导航</SheetTitle>
                  <SheetDescription>切换工作台和任务管理。</SheetDescription>
                </SheetHeader>
                <div className="flex flex-col gap-3 px-4">
                  <div className="flex flex-col gap-2">
                    {NAV_ITEMS.map((item) => {
                      const Icon = item.icon
                      return (
                        <NavLink
                          key={item.to}
                          to={item.to}
                          onClick={() => setMobileNavOpen(false)}
                        >
                          {({ isActive }) => (
                            <Button
                              variant={isActive ? "secondary" : "outline"}
                              className="w-full justify-start"
                            >
                              <Icon data-icon="inline-start" />
                              {item.label}
                            </Button>
                          )}
                        </NavLink>
                      )
                    })}
                  </div>
                </div>
              </SheetContent>
            </Sheet>
          </div>
        </header>

        <main className={cn("flex-1")}>{children}</main>
      </div>
    </div>
  )
}
