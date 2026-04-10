import { useState, type ReactNode } from "react"
import { Link, NavLink } from "react-router-dom"
import { FolderClock, LayoutDashboard, LaptopMinimal, Menu, MoonStar, SunMedium } from "lucide-react"

import type { DemoDashboard } from "@/hooks/use-demo-dashboard"
import { useThemeMode } from "@/hooks/use-theme-mode"
import type { ThemeMode } from "@/lib/theme"
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

const THEME_ITEMS: Array<{
  value: ThemeMode
  label: string
  icon: typeof SunMedium
}> = [
  { value: "light", label: "浅色", icon: SunMedium },
  { value: "dark", label: "深色", icon: MoonStar },
  { value: "system", label: "系统", icon: LaptopMinimal },
]

function nextThemeMode(currentMode: ThemeMode): ThemeMode {
  if (currentMode === "light") {
    return "dark"
  }
  if (currentMode === "dark") {
    return "system"
  }
  return "light"
}

function themeMeta(mode: ThemeMode) {
  return THEME_ITEMS.find((item) => item.value === mode) ?? THEME_ITEMS[0]
}

function ThemeModeButton({
  value,
  onValueChange,
}: {
  value: ThemeMode
  onValueChange: (value: ThemeMode) => void
}) {
  const current = themeMeta(value)
  const next = themeMeta(nextThemeMode(value))
  const Icon = current.icon

  return (
    <Button
      variant="outline"
      size="icon-sm"
      className="bg-background/70 backdrop-blur-sm"
      title={`当前${current.label}模式，点击切换到${next.label}`}
      onClick={() => onValueChange(next.value)}
    >
      <Icon />
      <span className="sr-only">{`当前${current.label}模式，点击切换到${next.label}`}</span>
    </Button>
  )
}

export function AppShell({
  dashboard,
  children,
}: {
  dashboard: DemoDashboard
  children: ReactNode
}) {
  const [mobileNavOpen, setMobileNavOpen] = useState(false)
  const theme = useThemeMode()

  return (
    <div className="min-h-screen bg-background">
      <a
        href="#main-content"
        className="sr-only absolute left-4 top-4 z-50 rounded-md bg-background px-3 py-2 text-sm shadow focus:not-sr-only"
      >
        跳到主要内容
      </a>

      <div className="mx-auto flex min-h-screen max-w-7xl flex-col gap-4 px-4 py-5 lg:px-6">
        <header className="flex items-center justify-between rounded-xl border border-border/70 bg-card/80 px-4 py-3 shadow-sm backdrop-blur-xl supports-[backdrop-filter]:bg-card/72">
          <div className="min-w-0">
            <Link to="/workspace" className="flex min-w-0 flex-col">
              <span className="text-xs tracking-[0.18em] text-muted-foreground uppercase" translate="no">
                PV-IQA
              </span>
              <span className="truncate text-base font-semibold tracking-tight">掌静脉质量评分</span>
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
            <ThemeModeButton value={theme.mode} onValueChange={theme.setMode} />
          </div>

          <div className="flex items-center gap-2 md:hidden">
            {dashboard.activeRunningCount ? (
              <Badge variant="outline">{dashboard.activeRunningCount} 处理中</Badge>
            ) : null}
            <ThemeModeButton value={theme.mode} onValueChange={theme.setMode} />
            <Sheet open={mobileNavOpen} onOpenChange={setMobileNavOpen}>
              <SheetTrigger asChild>
                <Button variant="ghost" size="icon-sm" aria-label="打开导航菜单">
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

        <main id="main-content" className={cn("flex-1")}>
          {children}
        </main>
      </div>
    </div>
  )
}
