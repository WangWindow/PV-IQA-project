import { useCallback, useEffect, useState } from "react"
import { Download, Filter, Search, Server, Shield, User as UserIcon } from "lucide-react"

import { useAuth } from "@/hooks/use-auth"
import type { AuditLog, AuditLogPage } from "@/lib/types"
import { exportLogs, fetchLogs } from "@/lib/api"
import { cn, downloadCsv } from "@/lib/utils"
import { formatTime } from "@/lib/format"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"

type LogFilter = "all" | "login" | "create" | "delete" | "update" | "stop_job" | "read"

const ACTION_COLORS: Record<string, string> = {
  login: "border-emerald-500/40 bg-emerald-50 text-emerald-700 dark:bg-emerald-950/30 dark:text-emerald-300",
  create: "border-sky-500/40 bg-sky-50 text-sky-700 dark:bg-sky-950/30 dark:text-sky-300",
  delete: "border-red-500/40 bg-red-50 text-red-700 dark:bg-red-950/30 dark:text-red-300",
  update: "border-amber-500/40 bg-amber-50 text-amber-700 dark:bg-amber-950/30 dark:text-amber-300",
  read: "border-violet-500/40 bg-violet-50 text-violet-700 dark:bg-violet-950/30 dark:text-violet-300",
  stop_job: "border-orange-500/40 bg-orange-50 text-orange-700 dark:bg-orange-950/30 dark:text-orange-300",
}

const FILTER_OPTIONS: Array<{ value: LogFilter; label: string; icon: typeof Shield }> = [
  { value: "all", label: "全部", icon: Filter },
  { value: "login", label: "登录", icon: UserIcon },
  { value: "create", label: "创建", icon: Server },
  { value: "read", label: "读取", icon: Search },
  { value: "update", label: "更新", icon: Server },
  { value: "delete", label: "删除", icon: Shield },
  { value: "stop_job", label: "停止", icon: Filter },
]

export function LogsPage() {
  const { user } = useAuth()
  const [logs, setLogs] = useState<AuditLog[]>([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(1)
  const [loading, setLoading] = useState(false)
  const [filter, setFilter] = useState<LogFilter>("all")
  const [searchUser, setSearchUser] = useState("")

  const pageSize = 20

  const loadLogs = useCallback(async () => {
    setLoading(true)
    try {
      const result: AuditLogPage = await fetchLogs({
        action: filter === "all" ? undefined : filter,
        userId: searchUser || undefined,
        page,
        pageSize,
      })
      setLogs(result.logs)
      setTotal(result.pagination.total)
    } catch {
      // 静默处理错误
    } finally {
      setLoading(false)
    }
  }, [filter, searchUser, page])

  useEffect(() => {
    void loadLogs()
  }, [loadLogs])

  const totalPages = Math.ceil(total / pageSize)

  async function handleExport() {
    try {
      const allLogs = await exportLogs()
      const headers = ["时间", "用户ID", "操作", "目标类型", "目标ID", "详情", "IP"]
      const rows = allLogs.map((log) => [
        formatTime(log.created_at),
        log.user_id || "—",
        log.action,
        log.target_type || "—",
        log.target_id || "—",
        log.detail || "—",
        log.ip_address || "—",
      ])
      downloadCsv("audit_logs.csv", headers, rows)
    } catch {
      // 导出失败
    }
  }

  const isAdminUser = user?.role === "admin"

  if (!isAdminUser) {
    return (
      <div className="flex min-h-[50vh] items-center justify-center">
        <span className="text-muted-foreground animate-infrared-pulse">仅管理员可查看操作日志</span>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      {/* Page title with accent */}
      <div className="flex items-center justify-between animate-fade-in-up">
        <div className="space-y-2">
          <h1 className="text-lg font-semibold">审计日志</h1>
          <div className="h-0.5 w-12 rounded-full bg-primary" />
        </div>
        <Button variant="outline" size="sm" onClick={handleExport} className="gap-1.5">
          <Download className="size-3.5" />
          导出 CSV
        </Button>
      </div>

      {/* 筛选栏 */}
      <Card className="animate-fade-in-up" style={{ animationDelay: "50ms" }}>
        <CardContent className="py-3">
          <div className="flex flex-wrap items-center gap-3">
            <ToggleGroup
              type="single"
              value={filter}
              onValueChange={(val) => {
                if (val) { setFilter(val as LogFilter); setPage(1) }
              }}
              variant="outline"
              size="sm"
              className="flex-wrap"
            >
              {FILTER_OPTIONS.map(({ value, label, icon: Icon }) => (
                <ToggleGroupItem key={value} value={value} className="gap-1.5 text-xs">
                  <Icon className="size-3" />
                  {label}
                </ToggleGroupItem>
              ))}
            </ToggleGroup>
            <div className="ml-auto">
              <Input
                placeholder="搜索用户 ID..."
                value={searchUser}
                onChange={(e) => { setSearchUser(e.target.value); setPage(1) }}
                className="h-8 w-48 text-xs"
              />
            </div>
          </div>
        </CardContent>
      </Card>

      {/* 日志表格 */}
      <Card className="animate-fade-in-up transition-shadow duration-300 hover:infrared-glow" style={{ animationDelay: "100ms" }}>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">审计日志</CardTitle>
          <CardDescription>
            共 <span className="stat-value text-sm">{total}</span> 条记录
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="w-[160px]">时间</TableHead>
                <TableHead className="w-[100px]">操作</TableHead>
                <TableHead className="w-[80px]">类型</TableHead>
                <TableHead>详情</TableHead>
                <TableHead className="w-[120px]">IP</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {loading ? (
                <TableRow>
                  <TableCell colSpan={5} className="py-8 text-center text-muted-foreground">
                    加载中...
                  </TableCell>
                </TableRow>
              ) : logs.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={5} className="py-8 text-center text-muted-foreground">
                    无日志记录
                  </TableCell>
                </TableRow>
              ) : (
                logs.map((log) => (
                  <TableRow key={log.id} className="even:bg-muted/30">
                    <TableCell className="font-data text-xs">{formatTime(log.created_at)}</TableCell>
                    <TableCell>
                      <Badge
                        variant="outline"
                        className={cn("text-xs font-medium", ACTION_COLORS[log.action] ?? "")}
                      >
                        {log.action}
                      </Badge>
                    </TableCell>
                    <TableCell className="text-xs text-muted-foreground">{log.target_type ?? "—"}</TableCell>
                    <TableCell className="max-w-[300px] truncate text-xs">{log.detail ?? "—"}</TableCell>
                    <TableCell className="font-data text-xs text-muted-foreground">{log.ip_address ?? "—"}</TableCell>
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>

          {/* 分页 */}
          {totalPages > 1 && (
            <div className="mt-4 flex items-center justify-center gap-2">
              <Button
                variant="outline"
                size="sm"
                disabled={page <= 1}
                onClick={() => setPage((p) => Math.max(1, p - 1))}
              >
                上一页
              </Button>
              <div className="flex items-center gap-1">
                {Array.from({ length: totalPages }, (_, i) => i + 1).map((p) => (
                  <Button
                    key={p}
                    variant={p === page ? "default" : "outline"}
                    size="sm"
                    className={cn(
                      "min-w-[2rem] px-2 text-xs",
                      p === page && "shadow-[0_0_12px_var(--primary)/0.25]"
                    )}
                    onClick={() => setPage(p)}
                  >
                    {p}
                  </Button>
                ))}
              </div>
              <Button
                variant="outline"
                size="sm"
                disabled={page >= totalPages}
                onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
              >
                下一页
              </Button>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
