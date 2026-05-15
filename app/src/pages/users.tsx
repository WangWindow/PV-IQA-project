import { useCallback, useEffect, useState } from "react"
import { Plus, ScanLine, Shield, Trash2, Users as UsersIcon } from "lucide-react"

import { useAuth } from "@/hooks/use-auth"
import type { UserInfo } from "@/lib/types"
import { adminCreateUser, deleteUser, listUsers, updateUserRole } from "@/lib/api"
import { formatTime } from "@/lib/format"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Empty, EmptyDescription, EmptyMedia, EmptyTitle } from "@/components/ui/empty"
import { Input } from "@/components/ui/input"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"

type UserWithCount = UserInfo & { job_count?: number }

export function UsersPage() {
  const { user: currentUser } = useAuth()
  const [users, setUsers] = useState<UserWithCount[]>([])
  const [loading, setLoading] = useState(false)
  const [createOpen, setCreateOpen] = useState(false)
  const [newUsername, setNewUsername] = useState("")
  const [newPassword, setNewPassword] = useState("")
  const [newRole, setNewRole] = useState<"admin" | "user">("user")
  const [creating, setCreating] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const loadUsers = useCallback(async () => {
    setLoading(true)
    try {
      const result = await listUsers()
      setUsers(result as UserWithCount[])
    } catch {
      // 静默
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void loadUsers()
  }, [loadUsers])

  async function handleCreateUser() {
    if (!newUsername || !newPassword) return
    setCreating(true)
    setError(null)
    try {
      await adminCreateUser(newUsername, newPassword, newRole)
      setCreateOpen(false)
      setNewUsername("")
      setNewPassword("")
      setNewRole("user")
      await loadUsers()
    } catch (err) {
      setError(err instanceof Error ? err.message : "创建失败")
    } finally {
      setCreating(false)
    }
  }

  async function handleDeleteUser(userId: string, username: string) {
    if (!confirm(`确定要删除用户「${username}」吗？此操作不可撤销。`)) return
    try {
      await deleteUser(userId)
      await loadUsers()
    } catch {
      // 静默
    }
  }

  async function handleToggleRole(userId: string, currentRole: string) {
    const newRoleVal = currentRole === "admin" ? "user" : "admin"
    try {
      await updateUserRole(userId, newRoleVal as "admin" | "user")
      await loadUsers()
    } catch {
      // 静默
    }
  }

  return (
    <div className="space-y-4">
      {/* Page title with accent */}
      <div className="flex items-center justify-between animate-fade-in-up">
        <div className="space-y-2">
          <h1 className="text-lg font-semibold">用户管理</h1>
          <div className="h-0.5 w-12 rounded-full bg-primary" />
        </div>
        <Button size="sm" onClick={() => setCreateOpen(true)}>
          <Plus className="mr-1.5 size-3.5" />
          创建用户
        </Button>
      </div>

      {error && (
        <Alert variant="destructive" className="animate-fade-in-up border-l-4 border-l-destructive">
          <AlertTitle>创建失败</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      <Card className="transition-shadow duration-300 hover:infrared-glow animate-fade-in-up" style={{ animationDelay: "50ms" }}>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">
            <UsersIcon className="size-4 text-primary" />
            用户列表
          </CardTitle>
          <CardDescription>
            共 <span className="stat-value text-sm">{users.length}</span> 个用户
          </CardDescription>
        </CardHeader>
        <CardContent>
          {loading ? (
            <div className="py-8 text-center">
              <span className="text-muted-foreground text-sm animate-infrared-pulse">加载中...</span>
            </div>
          ) : users.length === 0 ? (
            <Empty className="py-10">
              <EmptyMedia variant="icon">
                <ScanLine className="size-5 text-muted-foreground animate-scan-line" />
              </EmptyMedia>
              <EmptyTitle>暂无用户</EmptyTitle>
              <EmptyDescription>点击右上角「创建用户」添加新用户</EmptyDescription>
            </Empty>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>用户名</TableHead>
                  <TableHead>角色</TableHead>
                  <TableHead>任务数</TableHead>
                  <TableHead>注册时间</TableHead>
                  <TableHead className="text-right">操作</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {users.map((u) => (
                  <TableRow key={u.id} className="transition-colors hover:bg-accent/50">
                    <TableCell className="font-medium">{u.username}</TableCell>
                    <TableCell>
                      {u.role === "admin" ? (
                        <Badge className="gap-1 infrared-glow">
                          <Shield className="size-3" />
                          管理员
                        </Badge>
                      ) : (
                        <Badge variant="secondary">
                          普通用户
                        </Badge>
                      )}
                    </TableCell>
                    <TableCell>
                      <span className="stat-value text-sm">{u.job_count ?? 0}</span>
                    </TableCell>
                    <TableCell className="font-data text-xs text-muted-foreground">
                      {formatTime(u.created_at ?? null)}
                    </TableCell>
                    <TableCell className="text-right">
                      {u.id !== currentUser?.id && (
                        <div className="flex items-center justify-end gap-1">
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => handleToggleRole(u.id, u.role)}
                            title={u.role === "admin" ? "降为普通用户" : "升为管理员"}
                            className="transition-shadow hover:shadow-[0_0_12px_var(--primary)/0.2]"
                          >
                            <Shield className="size-3" />
                          </Button>
                          <Button
                            variant="outline"
                            size="sm"
                            className="text-destructive transition-shadow hover:text-destructive hover:shadow-[0_0_12px_var(--destructive)/0.2]"
                            onClick={() => handleDeleteUser(u.id, u.username)}
                            title="删除用户"
                          >
                            <Trash2 className="size-3" />
                          </Button>
                        </div>
                      )}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>

      {/* 创建用户对话框 */}
      <Dialog open={createOpen} onOpenChange={setCreateOpen}>
        <DialogContent className="sm:max-w-md infrared-glow">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <UsersIcon className="size-4 text-primary" />
              创建新用户
            </DialogTitle>
            <DialogDescription>由管理员创建新用户账户</DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-2">
            <div className="space-y-2">
              <label className="text-sm font-medium">用户名</label>
              <Input
                value={newUsername}
                onChange={(e) => setNewUsername(e.target.value)}
                placeholder="至少 2 个字符"
                minLength={2}
              />
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">密码</label>
              <Input
                type="password"
                value={newPassword}
                onChange={(e) => setNewPassword(e.target.value)}
                placeholder="至少 6 个字符"
                minLength={6}
              />
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">角色</label>
              <div className="flex gap-2">
                <button
                  type="button"
                  className={`rounded-md border px-3 py-1.5 text-sm font-medium transition-all ${
                    newRole === "user"
                      ? "border-primary bg-primary/10 text-primary shadow-[0_0_8px_var(--primary)/0.15]"
                      : "border-border text-muted-foreground hover:text-foreground"
                  }`}
                  onClick={() => setNewRole("user")}
                >
                  普通用户
                </button>
                <button
                  type="button"
                  className={`rounded-md border px-3 py-1.5 text-sm font-medium transition-all ${
                    newRole === "admin"
                      ? "border-primary bg-primary/10 text-primary shadow-[0_0_8px_var(--primary)/0.15]"
                      : "border-border text-muted-foreground hover:text-foreground"
                  }`}
                  onClick={() => setNewRole("admin")}
                >
                  管理员
                </button>
              </div>
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setCreateOpen(false)}>
              取消
            </Button>
            <Button onClick={handleCreateUser} disabled={creating || !newUsername || !newPassword}>
              {creating ? "创建中..." : "创建"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}
