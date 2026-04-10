export type ThemeMode = "light" | "dark" | "system"
export type ResolvedThemeMode = "light" | "dark"

export const THEME_MODE_STORAGE_KEY = "pv-iqa-theme-mode:v1"

export function isThemeMode(value: string | null | undefined): value is ThemeMode {
  return value === "light" || value === "dark" || value === "system"
}

export function readStoredThemeMode(): ThemeMode {
  if (typeof window === "undefined") {
    return "system"
  }
  try {
    const value = window.localStorage.getItem(THEME_MODE_STORAGE_KEY)
    return isThemeMode(value) ? value : "system"
  } catch {
    return "system"
  }
}

export function writeStoredThemeMode(mode: ThemeMode): void {
  if (typeof window === "undefined") {
    return
  }
  try {
    window.localStorage.setItem(THEME_MODE_STORAGE_KEY, mode)
  } catch {
    // Ignore transient storage failures and keep theme state in memory.
  }
}

export function systemThemeMode(): ResolvedThemeMode {
  if (typeof window === "undefined") {
    return "light"
  }
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"
}

export function resolveThemeMode(
  mode: ThemeMode,
  systemMode: ResolvedThemeMode = systemThemeMode()
): ResolvedThemeMode {
  return mode === "system" ? systemMode : mode
}

export function applyThemeMode(
  mode: ThemeMode,
  systemMode: ResolvedThemeMode = systemThemeMode()
): ResolvedThemeMode {
  const resolvedMode = resolveThemeMode(mode, systemMode)
  if (typeof document === "undefined") {
    return resolvedMode
  }

  const root = document.documentElement
  root.classList.toggle("dark", resolvedMode === "dark")
  root.style.colorScheme = resolvedMode
  root.dataset.theme = resolvedMode
  root.dataset.themeMode = mode
  return resolvedMode
}

export function bootstrapThemeMode(): ThemeMode {
  const mode = readStoredThemeMode()
  applyThemeMode(mode)
  return mode
}
