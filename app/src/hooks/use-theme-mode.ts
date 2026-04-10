import { useEffect, useMemo, useState } from "react"

import {
  applyThemeMode,
  readStoredThemeMode,
  resolveThemeMode,
  systemThemeMode,
  writeStoredThemeMode,
  type ThemeMode,
} from "@/lib/theme"

export function useThemeMode() {
  const [mode, setMode] = useState<ThemeMode>(() => readStoredThemeMode())
  const [systemMode, setSystemMode] = useState(() => systemThemeMode())

  useEffect(() => {
    const mediaQuery = window.matchMedia("(prefers-color-scheme: dark)")

    const updateSystemMode = (matches: boolean) => {
      setSystemMode(matches ? "dark" : "light")
    }

    updateSystemMode(mediaQuery.matches)

    const handleChange = (event: MediaQueryListEvent) => {
      updateSystemMode(event.matches)
    }

    if (typeof mediaQuery.addEventListener === "function") {
      mediaQuery.addEventListener("change", handleChange)
      return () => mediaQuery.removeEventListener("change", handleChange)
    }

    mediaQuery.addListener(handleChange)
    return () => mediaQuery.removeListener(handleChange)
  }, [])

  useEffect(() => {
    writeStoredThemeMode(mode)
    applyThemeMode(mode, systemMode)
  }, [mode, systemMode])

  const resolvedMode = useMemo(() => resolveThemeMode(mode, systemMode), [mode, systemMode])

  return {
    mode,
    resolvedMode,
    setMode,
  }
}
