import { useState, type FormEvent } from "react"
import { Link, useNavigate, useSearchParams } from "react-router-dom"
import { Eye, EyeOff } from "lucide-react"

import { useAuth } from "@/hooks/use-auth"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"

function PalmVeinLogo() {
  return (
    <svg
      viewBox="0 0 48 48"
      className="size-12 text-primary"
      fill="none"
      stroke="currentColor"
    >
      {/* Outer ring — palm boundary */}
      <circle cx="24" cy="24" r="20" strokeWidth="1.2" opacity="0.35" />
      {/* Scanning circle — NIR illumination area */}
      <circle cx="24" cy="24" r="14" strokeWidth="1" opacity="0.55" />
      {/* Vein pattern arcs */}
      <path
        d="M17 15 Q23 20 19 26 Q16 30 22 34"
        strokeWidth="1.3"
        strokeLinecap="round"
      />
      <path
        d="M27 14 Q31 18 29 24 Q27 30 33 34"
        strokeWidth="1.3"
        strokeLinecap="round"
      />
      <path
        d="M13 22 Q17 24 19 28"
        strokeWidth="0.9"
        strokeLinecap="round"
        opacity="0.7"
      />
      <path
        d="M35 20 Q31 24 29 28"
        strokeWidth="0.9"
        strokeLinecap="round"
        opacity="0.7"
      />
      {/* Center scan dot */}
      <circle cx="24" cy="24" r="2.5" fill="currentColor" stroke="none" opacity="0.85" />
      {/* Dashed scan ring */}
      <circle
        cx="24"
        cy="24"
        r="5"
        strokeWidth="0.6"
        opacity="0.3"
        strokeDasharray="3 2"
      />
      {/* Scan crosshairs */}
      <path d="M24 5v4" strokeWidth="0.6" opacity="0.25" />
      <path d="M24 39v4" strokeWidth="0.6" opacity="0.25" />
      <path d="M5 24h4" strokeWidth="0.6" opacity="0.25" />
      <path d="M39 24h4" strokeWidth="0.6" opacity="0.25" />
    </svg>
  )
}

export function LoginPage() {
  const { login } = useAuth()
  const navigate = useNavigate()
  const [searchParams] = useSearchParams()
  const registered = searchParams.get("registered") === "1"
  const [username, setUsername] = useState("")
  const [password, setPassword] = useState("")
  const [showPassword, setShowPassword] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  async function handleSubmit(e: FormEvent) {
    e.preventDefault()
    setError(null)
    setLoading(true)
    try {
      await login(username, password)
      navigate("/", { replace: true })
    } catch (err) {
      setError(err instanceof Error ? err.message : "登录失败")
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="relative flex min-h-screen items-center justify-center overflow-hidden">
      {/* ── Subtle infrared gradient background ─────────────── */}
      <div
        className="absolute inset-0"
        style={{
          background:
            "radial-gradient(ellipse at 15% 50%, var(--app-glow-left) 0%, transparent 55%)," +
            "radial-gradient(ellipse at 85% 50%, var(--app-glow-right) 0%, transparent 55%)," +
            "linear-gradient(180deg, var(--app-gradient-start), var(--app-gradient-end))",
        }}
      />

      {/* ── Content ─────────────────────────────────────────── */}
      <div className="relative z-10 w-full max-w-sm px-4 py-10">
        <div className="animate-fade-in-up">
          {/* Logo + Title */}
          <div className="mb-8 text-center">
            <div className="mx-auto mb-5 flex size-16 items-center justify-center rounded-2xl bg-card/80 border border-border/60 shadow-sm animate-infrared-pulse">
              <PalmVeinLogo />
            </div>
            <h1 className="text-2xl font-semibold tracking-tight text-infrared-gradient">
              掌静脉 IQA
            </h1>
            <p className="mt-1.5 text-sm text-muted-foreground">
              掌静脉图像质量评估系统
            </p>
          </div>

          {/* Auth Card */}
          <div className="infrared-glow rounded-xl border border-border bg-card px-6 pb-6 pt-5 shadow-lg">
            {/* Success banner */}
            {registered && (
              <div className="mb-5 rounded-lg border border-emerald-500/30 bg-emerald-50 px-3.5 py-2.5 text-sm font-medium text-emerald-700 dark:border-emerald-500/30 dark:bg-emerald-950/40 dark:text-emerald-300">
                <span className="flex items-center gap-2">
                  <svg
                    className="size-4 shrink-0"
                    viewBox="0 0 16 16"
                    fill="currentColor"
                  >
                    <path d="M8 15A7 7 0 1 0 8 1a7 7 0 0 0 0 14zm3.85-8.85L7.5 10.5 4.65 7.65a.5.5 0 0 1 .7-.7L7.5 9.3l3.65-3.64a.5.5 0 1 1 .7.7z" />
                  </svg>
                  注册成功！请使用新账户登录。
                </span>
              </div>
            )}

            <form onSubmit={handleSubmit} className="space-y-4">
              {/* Error state */}
              {error && (
                <div className="rounded-lg border border-destructive/30 bg-destructive/5 px-3.5 py-2.5 text-sm font-medium text-destructive dark:border-destructive/30 dark:bg-destructive/10 dark:text-destructive">
                  {error}
                </div>
              )}

              {/* Username */}
              <div className="space-y-1.5">
                <label
                  htmlFor="login-username"
                  className="text-sm font-medium text-foreground"
                >
                  用户名
                </label>
                <Input
                  id="login-username"
                  placeholder="请输入用户名"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  autoComplete="username"
                  required
                  className="h-10"
                />
              </div>

              {/* Password */}
              <div className="space-y-1.5">
                <label
                  htmlFor="login-password"
                  className="text-sm font-medium text-foreground"
                >
                  密码
                </label>
                <div className="relative">
                  <Input
                    id="login-password"
                    type={showPassword ? "text" : "password"}
                    placeholder="请输入密码"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    autoComplete="current-password"
                    required
                    className="h-10 pr-10"
                  />
                  <button
                    type="button"
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors"
                    onClick={() => setShowPassword((v) => !v)}
                    tabIndex={-1}
                  >
                    {showPassword ? (
                      <EyeOff className="size-4" />
                    ) : (
                      <Eye className="size-4" />
                    )}
                  </button>
                </div>
              </div>

              {/* Submit */}
              <Button
                type="submit"
                className="infrared-glow h-10 w-full font-semibold tracking-wide"
                disabled={loading}
              >
                {loading ? (
                  <span className="flex items-center gap-2">
                    <span className="size-3.5 animate-spin rounded-full border-2 border-current border-r-transparent" />
                    登录中...
                  </span>
                ) : (
                  "登录"
                )}
              </Button>
            </form>
          </div>

          {/* Register link */}
          <p className="mt-5 text-center text-sm text-muted-foreground">
            没有账户？{" "}
            <Link
              to="/register"
              className="font-semibold text-primary underline-offset-4 hover:underline"
            >
              注册
            </Link>
          </p>
        </div>
      </div>
    </div>
  )
}
