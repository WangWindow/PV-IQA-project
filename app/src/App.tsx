import { lazy, Suspense, type ReactNode } from "react"
import { AnimatePresence } from "motion/react"
import { Navigate, Route, Routes, useLocation } from "react-router-dom"

import { AuthProvider, useAuth } from "@/hooks/use-auth"
import { AppShell } from "@/components/app-shell"
import { useDashboard } from "@/hooks/use-dashboard"
import { Spinner } from "@/components/ui/spinner"

const WorkspacePage = lazy(async () => {
  const module = await import("@/pages/workspace")
  return { default: module.WorkspacePage }
})

const HistoryPage = lazy(async () => {
  const module = await import("@/pages/history")
  return { default: module.HistoryPage }
})

const LoginPage = lazy(async () => {
  const module = await import("@/pages/login")
  return { default: module.LoginPage }
})

const RegisterPage = lazy(async () => {
  const module = await import("@/pages/register")
  return { default: module.RegisterPage }
})

const LogsPage = lazy(async () => {
  const module = await import("@/pages/logs")
  return { default: module.LogsPage }
})

const SettingsPage = lazy(async () => {
  const module = await import("@/pages/settings")
  return { default: module.SettingsPage }
})

const UsersPage = lazy(async () => {
  const module = await import("@/pages/users")
  return { default: module.UsersPage }
})

function SuspenseFallback() {
  return (
    <div className="flex min-h-[50vh] items-center justify-center">
      <Spinner className="size-5" />
    </div>
  )
}

/** 认证守卫：未登录重定向到 /login */
function RequireAuth({ children }: { children: ReactNode }) {
  const { isAuthenticated, isLoading } = useAuth()

  if (isLoading) {
    return <SuspenseFallback />
  }

  if (!isAuthenticated) {
    return <Navigate to="/login" replace />
  }

  return <>{children}</>
}

/** 管理员守卫 */
function RequireAdmin({ children }: { children: ReactNode }) {
  const { user } = useAuth()

  if (user?.role !== "admin") {
    return <Navigate to="/workspace" replace />
  }

  return <>{children}</>
}

function AppRoutes() {
  const dashboard = useDashboard()
  const location = useLocation()

  return (
    <AnimatePresence mode="wait">
      <Routes location={location} key={location.pathname}>
        {/* 公开路由 */}
        <Route path="/login" element={<LoginPage />} />
        <Route path="/register" element={<RegisterPage />} />

        {/* 受保护路由 — 所有已认证用户 */}
        <Route
          path="/workspace"
          element={
            <RequireAuth>
              <AppShell dashboard={dashboard}>
                <Suspense fallback={<SuspenseFallback />}>
                  <WorkspacePage dashboard={dashboard} />
                </Suspense>
              </AppShell>
            </RequireAuth>
          }
        />
        <Route
          path="/jobs"
          element={
            <RequireAuth>
              <AppShell dashboard={dashboard}>
                <Suspense fallback={<SuspenseFallback />}>
                  <HistoryPage dashboard={dashboard} />
                </Suspense>
              </AppShell>
            </RequireAuth>
          }
        />
        <Route
          path="/settings"
          element={
            <RequireAuth>
              <AppShell dashboard={dashboard}>
                <Suspense fallback={<SuspenseFallback />}>
                  <SettingsPage />
                </Suspense>
              </AppShell>
            </RequireAuth>
          }
        />

        {/* 管理员专属路由 */}
        <Route
          path="/logs"
          element={
            <RequireAuth>
              <RequireAdmin>
                <AppShell dashboard={dashboard}>
                  <Suspense fallback={<SuspenseFallback />}>
                    <LogsPage />
                  </Suspense>
                </AppShell>
              </RequireAdmin>
            </RequireAuth>
          }
        />
        <Route
          path="/users"
          element={
            <RequireAuth>
              <RequireAdmin>
                <AppShell dashboard={dashboard}>
                  <Suspense fallback={<SuspenseFallback />}>
                    <UsersPage />
                  </Suspense>
                </AppShell>
              </RequireAdmin>
            </RequireAuth>
          }
        />

        {/* 默认重定向 */}
        <Route path="/" element={<Navigate to="/workspace" replace />} />
        <Route path="*" element={<Navigate to="/workspace" replace />} />
      </Routes>
    </AnimatePresence>
  )
}

function App() {
  return (
    <AuthProvider>
      <AppRoutes />
    </AuthProvider>
  )
}

export default App