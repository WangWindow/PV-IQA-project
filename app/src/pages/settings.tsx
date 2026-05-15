import { useCallback, useEffect, useState } from "react"
import { Database, Lock, RefreshCw, Save, Server } from "lucide-react"

import { useAuth, useIsAdmin } from "@/hooks/use-auth"
import type { SystemSettings } from "@/lib/types"
import { changePassword as changePasswordApi, fetchDbStats, fetchSettings, updateSettings } from "@/lib/api"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import {
  Field,
  FieldContent,
  FieldDescription,
  FieldGroup,
  FieldLabel,
} from "@/components/ui/field"
import { Input } from "@/components/ui/input"

export function SettingsPage() {
  const { user } = useAuth()
  const isAdmin = useIsAdmin()

  // 系统设置
  const [settings, setSettings] = useState<SystemSettings | null>(null)
  const [settingsLoading, setSettingsLoading] = useState(false)
  const [settingsSaving, setSettingsSaving] = useState(false)
  const [settingsMessage, setSettingsMessage] = useState<{ type: "success" | "error"; text: string } | null>(null)

  // 密码修改
  const [oldPassword, setOldPassword] = useState("")
  const [newPassword, setNewPassword] = useState("")
  const [confirmPassword, setConfirmPassword] = useState("")
  const [passwordSaving, setPasswordSaving] = useState(false)
  const [passwordMessage, setPasswordMessage] = useState<{ type: "success" | "error"; text: string } | null>(null)

  // 数据库统计
  const [dbStats, setDbStats] = useState<{ jobs: number; results: number; users: number; audit_logs: number } | null>(null)

  const loadSettings = useCallback(async () => {
    setSettingsLoading(true)
    try {
      const s = await fetchSettings()
      setSettings(s)
    } catch {
      // 静默
    } finally {
      setSettingsLoading(false)
    }
  }, [])

  const loadDbStats = useCallback(async () => {
    try {
      const result = await fetchDbStats()
      setDbStats(result.stats)
    } catch {
      // 静默
    }
  }, [])

  useEffect(() => {
    void loadSettings()
    void loadDbStats()
  }, [loadSettings, loadDbStats])

  async function handleSaveSettings() {
    if (!settings) return
    setSettingsSaving(true)
    setSettingsMessage(null)
    try {
      await updateSettings(settings)
      setSettingsMessage({ type: "success", text: "设置已保存" })
    } catch (err) {
      setSettingsMessage({ type: "error", text: err instanceof Error ? err.message : "保存失败" })
    } finally {
      setSettingsSaving(false)
    }
  }

  async function handlePasswordChange() {
    setPasswordMessage(null)
    if (newPassword !== confirmPassword) {
      setPasswordMessage({ type: "error", text: "两次输入的密码不一致" })
      return
    }
    if (newPassword.length < 6) {
      setPasswordMessage({ type: "error", text: "新密码长度至少为 6 个字符" })
      return
    }
    setPasswordSaving(true)
    try {
      await changePasswordApi(oldPassword, newPassword)
      setPasswordMessage({ type: "success", text: "密码修改成功" })
      setOldPassword("")
      setNewPassword("")
      setConfirmPassword("")
    } catch (err) {
      setPasswordMessage({ type: "error", text: err instanceof Error ? err.message : "修改失败" })
    } finally {
      setPasswordSaving(false)
    }
  }

  if (!settings && settingsLoading) {
    return (
      <div className="flex min-h-[50vh] items-center justify-center">
        <span className="text-muted-foreground animate-infrared-pulse">加载设置中...</span>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Page title with accent line */}
      <div className="space-y-2 animate-fade-in-up">
        <h1 className="text-lg font-semibold">系统设置</h1>
        <div className="h-0.5 w-12 rounded-full bg-primary" />
      </div>

      {/* 系统配置（仅管理员） */}
      {isAdmin && settings && (
        <Card
          className="transition-shadow duration-300 hover:infrared-glow animate-fade-in-up"
          style={{ animationDelay: "50ms" }}
        >
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <Server className="size-4 text-primary" />
              系统配置
            </CardTitle>
            <CardDescription>调整系统参数（仅管理员可修改）</CardDescription>
          </CardHeader>
          <CardContent>
            <FieldGroup className="space-y-5">
              <Field>
                <FieldLabel>默认推理后端</FieldLabel>
                <FieldContent>
                  <select
                    className="h-9 w-full rounded-md border bg-background px-3 text-sm transition-shadow focus:outline-none focus-visible:ring-2 focus-visible:ring-ring/50"
                    value={settings.default_backend}
                    onChange={(e) => setSettings({ ...settings, default_backend: e.target.value })}
                  >
                    <option value="python">Python (PyTorch)</option>
                    <option value="rust">Rust (ONNX)</option>
                  </select>
                </FieldContent>
                <FieldDescription>新建评分任务时默认选择的推理后端</FieldDescription>
              </Field>

              <Field>
                <FieldLabel>默认计算设备</FieldLabel>
                <FieldContent>
                  <select
                    className="h-9 w-full rounded-md border bg-background px-3 text-sm transition-shadow focus:outline-none focus-visible:ring-2 focus-visible:ring-ring/50"
                    value={settings.default_device}
                    onChange={(e) => setSettings({ ...settings, default_device: e.target.value })}
                  >
                    <option value="cpu">CPU</option>
                    <option value="cuda">CUDA (GPU)</option>
                  </select>
                </FieldContent>
                <FieldDescription>新建评分任务时默认选择的计算设备</FieldDescription>
              </Field>

              <Field>
                <FieldLabel>最大上传大小 (MB)</FieldLabel>
                <FieldContent>
                  <Input
                    type="number"
                    min="10"
                    max="1024"
                    value={settings.max_upload_size_mb}
                    onChange={(e) => setSettings({ ...settings, max_upload_size_mb: e.target.value })}
                  />
                </FieldContent>
                <FieldDescription>单个图片上传的最大文件大小限制</FieldDescription>
              </Field>

              <Field>
                <FieldLabel>系统名称</FieldLabel>
                <FieldContent>
                  <Input
                    value={settings.system_name}
                    onChange={(e) => setSettings({ ...settings, system_name: e.target.value })}
                  />
                </FieldContent>
                <FieldDescription>显示在浏览器标题栏的系统名称</FieldDescription>
              </Field>

              <Field>
                <FieldLabel>任务保留天数</FieldLabel>
                <FieldContent>
                  <Input
                    type="number"
                    min="7"
                    max="365"
                    value={settings.job_retention_days}
                    onChange={(e) => setSettings({ ...settings, job_retention_days: e.target.value })}
                  />
                </FieldContent>
                <FieldDescription>超过此天数的已完成任务可被自动清理</FieldDescription>
              </Field>

              {settingsMessage && (
                <Alert
                  variant={settingsMessage.type === "error" ? "destructive" : "default"}
                  className={
                    settingsMessage.type === "success"
                      ? "border-l-4 border-l-emerald-500 bg-emerald-50/50 dark:bg-emerald-950/20"
                      : "border-l-4 border-l-destructive"
                  }
                >
                  <AlertDescription>{settingsMessage.text}</AlertDescription>
                </Alert>
              )}

              <div className="flex gap-2 pt-1">
                <Button onClick={handleSaveSettings} disabled={settingsSaving}>
                  <Save className="mr-1.5 size-3.5" />
                  {settingsSaving ? "保存中..." : "保存设置"}
                </Button>
                <Button variant="outline" onClick={() => void loadSettings()}>
                  <RefreshCw className="mr-1.5 size-3.5" />
                  重置
                </Button>
              </div>
            </FieldGroup>
          </CardContent>
        </Card>
      )}

      {/* 数据库统计（仅管理员） */}
      {isAdmin && dbStats && (
        <Card
          className="transition-shadow duration-300 hover:infrared-glow animate-fade-in-up"
          style={{ animationDelay: "100ms" }}
        >
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <Database className="size-4 text-foreground/70" />
              数据库统计
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
              {[
                { label: "任务数", value: dbStats.jobs },
                { label: "结果数", value: dbStats.results },
                { label: "用户数", value: dbStats.users },
                { label: "日志条数", value: dbStats.audit_logs },
              ].map((stat, i) => (
                <div
                  key={stat.label}
                  className="group/stat rounded-xl border bg-card p-4 transition-all duration-300 hover:cyan-glow hover:scale-[1.02]"
                  style={{ animationDelay: `${120 + i * 60}ms` }}
                >
                  <div className="text-sm text-muted-foreground">{stat.label}</div>
                  <div className="stat-value mt-1 text-xl text-foreground">{stat.value}</div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* 账户设置 */}
      <Card
        className="transition-shadow duration-300 hover:infrared-glow animate-fade-in-up"
        style={{ animationDelay: "150ms" }}
      >
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">
            <Lock className="size-4 text-primary" />
            账户设置
          </CardTitle>
          <CardDescription>
            当前用户：
            <span className="font-medium text-foreground">{user?.username ?? "—"}</span>
            （{user?.role === "admin" ? "管理员" : "普通用户"}）
          </CardDescription>
        </CardHeader>
        <CardContent>
          <FieldGroup className="space-y-4">
            <Field>
              <FieldLabel>旧密码</FieldLabel>
              <FieldContent>
                <Input
                  type="password"
                  value={oldPassword}
                  onChange={(e) => setOldPassword(e.target.value)}
                  placeholder="输入当前密码"
                  autoComplete="current-password"
                />
              </FieldContent>
            </Field>

            <Field>
              <FieldLabel>新密码</FieldLabel>
              <FieldContent>
                <Input
                  type="password"
                  value={newPassword}
                  onChange={(e) => setNewPassword(e.target.value)}
                  placeholder="输入新密码（至少 6 个字符）"
                  autoComplete="new-password"
                />
              </FieldContent>
            </Field>

            <Field>
              <FieldLabel>确认新密码</FieldLabel>
              <FieldContent>
                <Input
                  type="password"
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  placeholder="再次输入新密码"
                  autoComplete="new-password"
                />
              </FieldContent>
            </Field>

            {passwordMessage && (
              <Alert
                variant={passwordMessage.type === "error" ? "destructive" : "default"}
                className={
                  passwordMessage.type === "success"
                    ? "border-l-4 border-l-emerald-500 bg-emerald-50/50 dark:bg-emerald-950/20"
                    : "border-l-4 border-l-destructive"
                }
              >
                <AlertDescription>{passwordMessage.text}</AlertDescription>
              </Alert>
            )}

            <Button onClick={handlePasswordChange} disabled={passwordSaving} variant="secondary" className="w-fit">
              {passwordSaving ? "修改中..." : "修改密码"}
            </Button>
          </FieldGroup>
        </CardContent>
      </Card>
    </div>
  )
}
