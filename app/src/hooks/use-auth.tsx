import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from "react"

import type { AuthState } from "@/lib/types"
import { changePassword as changePasswordApi, fetchCurrentUser, loginApi, registerApi } from "@/lib/api"

type AuthContextType = AuthState & {
  login: (username: string, password: string) => Promise<void>
  register: (username: string, password: string) => Promise<void>
  logout: () => void
  changePassword: (oldPassword: string, newPassword: string) => Promise<void>
}

const AUTH_TOKEN_KEY = "pv-iqa-token"

const AuthContext = createContext<AuthContextType | null>(null)

export function AuthProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<AuthState>({
    user: null,
    token: null,
    isAuthenticated: false,
    isLoading: true,
  })

  // 应用启动时恢复认证状态
  useEffect(() => {
    const token = localStorage.getItem(AUTH_TOKEN_KEY)
    if (!token) {
      setState((s) => ({ ...s, isLoading: false }))
      return
    }

    fetchCurrentUser()
      .then((user) => {
        if (user) {
          setState({ user, token, isAuthenticated: true, isLoading: false })
        } else {
          // 令牌无效，清除
          localStorage.removeItem(AUTH_TOKEN_KEY)
          setState({ user: null, token: null, isAuthenticated: false, isLoading: false })
        }
      })
      .catch(() => {
        localStorage.removeItem(AUTH_TOKEN_KEY)
        setState({ user: null, token: null, isAuthenticated: false, isLoading: false })
      })
  }, [])

  const login = useCallback(async (username: string, password: string) => {
    const response = await loginApi({ username, password })
    localStorage.setItem(AUTH_TOKEN_KEY, response.access_token)
    setState({
      user: response.user,
      token: response.access_token,
      isAuthenticated: true,
      isLoading: false,
    })
  }, [])

  const register = useCallback(async (username: string, password: string) => {
    // 注册仅创建账户，不自动登录。让用户跳转到登录页手动登录。
    await registerApi({ username, password })
  }, [])

  const logout = useCallback(() => {
    localStorage.removeItem(AUTH_TOKEN_KEY)
    setState({ user: null, token: null, isAuthenticated: false, isLoading: false })
  }, [])

  const changePasswordFn = useCallback(async (oldPassword: string, newPassword: string) => {
    await changePasswordApi(oldPassword, newPassword)
  }, [])

  return (
    <AuthContext.Provider value={{ ...state, login, register, logout, changePassword: changePasswordFn }}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth(): AuthContextType {
  const context = useContext(AuthContext)
  if (!context) {
    throw new Error("useAuth must be used within an AuthProvider")
  }
  return context
}

/** 判断当前用户是否为管理员 */
export function useIsAdmin(): boolean {
  const { user } = useAuth()
  return user?.role === "admin"
}