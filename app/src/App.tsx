import { lazy, Suspense } from "react"
import { AnimatePresence } from "motion/react"
import { Navigate, Route, Routes, useLocation } from "react-router-dom"

import { AppShell } from "@/components/demo/app-shell"
import { useDemoDashboard } from "@/hooks/use-demo-dashboard"
import { Spinner } from "@/components/ui/spinner"

const WorkspacePage = lazy(async () => {
  const module = await import("@/components/demo/workspace-page")
  return { default: module.WorkspacePage }
})

const HistoryPage = lazy(async () => {
  const module = await import("@/components/demo/history-page")
  return { default: module.HistoryPage }
})

function App() {
  const dashboard = useDemoDashboard()
  const location = useLocation()

  return (
    <AppShell dashboard={dashboard}>
      <Suspense
        fallback={
          <div className="flex min-h-[50vh] items-center justify-center">
            <Spinner className="size-5" />
          </div>
        }
      >
        <AnimatePresence mode="wait">
          <Routes location={location} key={location.pathname}>
            <Route path="/" element={<Navigate to="/workspace" replace />} />
            <Route path="/workspace" element={<WorkspacePage dashboard={dashboard} />} />
            <Route path="/jobs" element={<HistoryPage dashboard={dashboard} />} />
          </Routes>
        </AnimatePresence>
      </Suspense>
    </AppShell>
  )
}

export default App
