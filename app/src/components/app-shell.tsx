import { useState, type ReactNode } from "react"
import { Link, NavLink } from "react-router-dom"
import {
  FolderClock,
  LayoutDashboard,
  LaptopMinimal,
  LogOut,
  Menu,
  MoonStar,
  Settings,
  SunMedium,
  Users,
  FileText,
  KeyRound,
} from "lucide-react"

import type { Dashboard } from "@/hooks/use-dashboard"
import { useAuth } from "@/hooks/use-auth"
import { useThemeMode } from "@/hooks/use-theme-mode"
import type { ThemeMode } from "@/lib/theme"
import { cn } from "@/lib/utils"
import {
  Sheet,
  SheetClose,
  SheetContent,
  SheetTrigger,
} from "@/components/ui/sheet"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"

const NAV_ITEMS_USER = [
  { to: "/workspace", label: "工作台", icon: LayoutDashboard },
  { to: "/jobs", label: "我的任务", icon: FolderClock },
  { to: "/settings", label: "设置", icon: Settings },
] as const

const NAV_ITEMS_ADMIN = [
  { to: "/workspace", label: "工作台", icon: LayoutDashboard },
  { to: "/jobs", label: "所有任务", icon: FolderClock },
  { to: "/users", label: "用户", icon: Users },
  { to: "/logs", label: "日志", icon: FileText },
  { to: "/settings", label: "设置", icon: Settings },
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
  if (currentMode === "light") return "dark"
  if (currentMode === "dark") return "system"
  return "light"
}

function themeMeta(mode: ThemeMode) {
  return THEME_ITEMS.find((item) => item.value === mode) ?? THEME_ITEMS[0]
}

/** PV-IQA Logo — 掌静脉纹路图标 */
function AppLogo({ className }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 24 24"
      className={className}
      fill="none"
      stroke="currentColor"
      strokeWidth={1.8}
    >
      <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2z" />
      <path
        d="M8 12c0-2.21 1.79-4 4-4s4 1.79 4 4-1.79 4-4 4"
        strokeLinecap="round"
      />
      <circle cx="12" cy="12" r="1.5" fill="currentColor" stroke="none" />
    </svg>
  )
}

export function AppShell({ children }: { dashboard: Dashboard; children: ReactNode }) {
  const [mobileNavOpen, setMobileNavOpen] = useState(false)
  const theme = useThemeMode()
  const { user, logout, isAuthenticated } = useAuth()
  const navItems = user?.role === "admin" ? NAV_ITEMS_ADMIN : NAV_ITEMS_USER

  // ── 共享侧边栏内容（桌面 & 移动复用） ──────────────────────
  function SidebarContent({ onNavClick }: { onNavClick?: () => void }) {
    return (
      <div className="flex h-full flex-col">
        {/* ── Logo 区域 ──────────────────────────────────── */}
        <Link
          to="/workspace"
          onClick={onNavClick}
          className="group flex items-center gap-2.5 px-4 pt-5 pb-4"
        >
          <AppLogo className="size-7 shrink-0 text-[var(--sidebar-primary)] transition-transform group-hover:scale-110" />
          <span className="text-lg font-semibold tracking-tight text-[var(--sidebar-foreground)]">
            PV-IQA
          </span>
        </Link>

        {/* ── 导航菜单 ──────────────────────────────────── */}
        <nav className="flex-1 space-y-0.5 px-3" role="navigation" aria-label="主导航">
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              onClick={onNavClick}
              className={({ isActive }) =>
                cn(
                  "relative flex items-center gap-3 rounded-r-md px-3 py-2.5 text-sm font-medium transition-all duration-200",
                  "border-l-[3px] border-transparent",
                  isActive
                    ? [
                        "border-[var(--sidebar-primary)]",
                        "bg-[var(--sidebar-primary)]/10",
                        "text-[var(--sidebar-foreground)] font-semibold",
                        "shadow-[2px_0_12px_-2px_var(--primary)_/_0.15]",
                        "dark:shadow-[2px_0_16px_-2px_var(--sidebar-primary)_/_0.25]",
                      ]
                    : [
                        "text-[var(--sidebar-foreground)]/65",
                        "hover:bg-[var(--sidebar-accent)]/50",
                        "hover:text-[var(--sidebar-foreground)]",
                        "hover:border-[var(--sidebar-accent)]/30",
                      ]
                )
              }
            >
              <item.icon
                className="size-4 shrink-0"
                aria-hidden="true"
              />
              {item.label}
            </NavLink>
          ))}
        </nav>

        {/* ── 底部：主题切换 & 用户 ──────────────────────── */}
        <div className="border-t border-[var(--sidebar-border)] px-3 py-3 space-y-2">
          {/* 主题循环按钮 */}
          <ThemeToggle
            value={theme.mode}
            onValueChange={theme.setMode}
          />

          {/* 用户信息 & 下拉菜单 */}
          {isAuthenticated && user ? (
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <button
                  type="button"
                  className="flex w-full items-center gap-2.5 rounded-md px-2 py-1.5 text-left transition-colors hover:bg-[var(--sidebar-accent)]/40"
                  title={user.username}
                >
                  <div className="flex size-8 shrink-0 items-center justify-center rounded-full bg-[var(--sidebar-primary)] text-xs font-semibold text-[var(--sidebar-primary-foreground)]">
                    {user.username.charAt(0).toUpperCase()}
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="truncate text-sm font-medium text-[var(--sidebar-foreground)]">
                      {user.username}
                    </div>
                    <div className="text-xs text-[var(--sidebar-foreground)]/50">
                      {user.role === "admin" ? "管理员" : "普通用户"}
                    </div>
                  </div>
                </button>
              </DropdownMenuTrigger>
              <DropdownMenuContent
                align="end"
                side="right"
                sideOffset={8}
                className="w-44"
              >
                <DropdownMenuItem asChild>
                  <Link
                    to="/settings"
                    className="flex items-center gap-2"
                  >
                    <KeyRound className="size-3.5" />
                    修改密码
                  </Link>
                </DropdownMenuItem>
                <DropdownMenuSeparator />
                <DropdownMenuItem
                  onClick={logout}
                  className="text-destructive focus:text-destructive"
                >
                  <LogOut className="mr-2 size-3.5" />
                  登出
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          ) : null}
        </div>
      </div>
    )
  }

  // ── 渲染 ──────────────────────────────────────────────────
  return (
    <div className="min-h-screen bg-background">
      {/* 无障碍跳转链接 */}
      <a
        href="#main-content"
        className="sr-only absolute left-4 top-4 z-50 rounded-md bg-background px-3 py-2 text-sm shadow focus:not-sr-only"
      >
        跳到主要内容
      </a>

      {/* ── 桌面端固定侧边栏 ────────────────────────────── */}
      <aside
        className="hidden lg:flex lg:flex-col lg:fixed lg:inset-y-0 lg:left-0 lg:z-40 lg:w-60 lg:border-r lg:border-[var(--sidebar-border)] lg:bg-[var(--sidebar)]"
        role="complementary"
        aria-label="侧边导航"
      >
        <SidebarContent />

        {/* 深色模式：侧边栏右边缘红外辉光 */}
        <div
          className="pointer-events-none absolute inset-y-0 -right-px w-px bg-gradient-to-b from-transparent via-[var(--sidebar-primary)]/20 to-transparent dark:opacity-100 opacity-0 transition-opacity duration-500"
          aria-hidden="true"
        />
        <div
          className="pointer-events-none absolute inset-y-0 right-0 w-4 bg-gradient-to-l from-[var(--sidebar-primary)]/[0.04] to-transparent dark:opacity-100 opacity-0 transition-opacity duration-500"
          aria-hidden="true"
        />
      </aside>

      {/* ── 移动端顶部栏 + Sheet ────────────────────────── */}
      <header className="lg:hidden sticky top-0 z-40 flex items-center h-12 px-3 bg-background/95 backdrop-blur-md border-b border-border">
        <Sheet open={mobileNavOpen} onOpenChange={setMobileNavOpen}>
          <SheetTrigger asChild>
            <button
              type="button"
              className="rounded-md p-1.5 text-muted-foreground transition-colors hover:text-foreground hover:bg-muted"
              aria-label="打开导航菜单"
            >
              <Menu className="size-5" />
            </button>
          </SheetTrigger>
          <SheetContent
            side="left"
            showCloseButton={false}
            className="w-72 p-0 bg-[var(--sidebar)] border-r-[var(--sidebar-border)]"
          >
            {/* 移动 Sheet 顶部关闭区域 */}
            <div className="flex items-center justify-between px-3 pt-3 pb-1">
              <SheetClose asChild>
                <button
                  type="button"
                  className="rounded-md p-1.5 text-[var(--sidebar-foreground)]/60 transition-colors hover:text-[var(--sidebar-foreground)] hover:bg-[var(--sidebar-accent)]/40"
                  aria-label="关闭导航菜单"
                >
                  <svg
                    width="15"
                    height="15"
                    viewBox="0 0 15 15"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="1.5"
                  >
                    <path d="M11.7816 4.03157C12.0062 3.80702 12.0062 3.44295 11.7816 3.2184C11.5571 2.99385 11.193 2.99385 10.9685 3.2184L7.50005 6.68682L4.03164 3.2184C3.80708 2.99385 3.44301 2.99385 3.21846 3.2184C2.99391 3.44295 2.99391 3.80702 3.21846 4.03157L6.68688 7.49999L3.21846 10.9684C2.99391 11.193 2.99391 11.557 3.21846 11.7816C3.44301 12.0061 3.80708 12.0061 4.03164 11.7816L7.50005 8.31316L10.9685 11.7816C11.193 12.0061 11.5571 12.0061 11.7816 11.7816C12.0062 11.557 12.0062 11.193 11.7816 10.9684L8.31322 7.49999L11.7816 4.03157Z" />
                  </svg>
                </button>
              </SheetClose>
            </div>
            <SidebarContent onNavClick={() => setMobileNavOpen(false)} />
          </SheetContent>
        </Sheet>

        {/* 移动端居中 Logo */}
        <Link
          to="/workspace"
          className="absolute left-1/2 -translate-x-1/2 flex items-center gap-1.5 text-foreground"
        >
          <AppLogo className="size-5 text-primary" />
          <span className="text-sm font-semibold tracking-tight">PV-IQA</span>
        </Link>

        {/* 移动端右侧快捷操作 */}
        <div className="flex-1" />
        {isAuthenticated && user ? (
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <button
                type="button"
                className="flex size-7 items-center justify-center rounded-full bg-primary/10 text-xs font-semibold text-primary transition-colors hover:bg-primary/20"
                title={user.username}
              >
                {user.username.charAt(0).toUpperCase()}
              </button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-44">
              <DropdownMenuItem asChild>
                <Link to="/settings" className="flex items-center gap-2">
                  <KeyRound className="size-3.5" />
                  修改密码
                </Link>
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem
                onClick={logout}
                className="text-destructive focus:text-destructive"
              >
                <LogOut className="mr-2 size-3.5" />
                登出
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        ) : null}
      </header>

      {/* ── 主内容区域 ──────────────────────────────────── */}
      <main
        id="main-content"
        className="lg:pl-60"
      >
        <div className="mx-auto max-w-7xl px-4 py-5 lg:px-6 lg:py-6">
          {children}
        </div>
      </main>
    </div>
  )
}

/** 主题循环切换按钮（浅色 → 深色 → 系统） */
function ThemeToggle({
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
    <button
      type="button"
      className={cn(
        "flex w-full items-center gap-2.5 rounded-md px-2 py-1.5 text-sm font-medium transition-colors",
        "text-[var(--sidebar-foreground)]/65",
        "hover:bg-[var(--sidebar-accent)]/40 hover:text-[var(--sidebar-foreground)]"
      )}
      title={`当前${current.label}模式，点击切换到${next.label}`}
      onClick={() => onValueChange(next.value)}
    >
      <Icon className="size-4 shrink-0" />
      <span>{current.label}模式</span>
    </button>
  )
}
