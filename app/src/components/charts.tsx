import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts"
import type { JobTrendPoint, RankedScorePoint, ScoreBandPoint } from "@/lib/analytics"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"

function EmptyChart({ description }: { description: string }) {
  return (
    <div className="relative flex h-[220px] items-center justify-center overflow-hidden rounded-xl border border-dashed border-border bg-muted/20 px-6 text-center">
      {/* Scan-line decorative element */}
      <div
        className="absolute inset-0 opacity-[0.03]"
        style={{
          backgroundImage:
            "repeating-linear-gradient(0deg, var(--primary), var(--primary) 1px, transparent 1px, transparent 4px)",
        }}
      />
      <span className="relative text-sm text-muted-foreground">{description}</span>
    </div>
  )
}

const tooltipStyle = {
  borderRadius: 12,
  border: "1px solid var(--primary)",
  background: "var(--card)",
  color: "var(--card-foreground)",
  boxShadow:
    "0 4px 24px oklch(0.55 0.22 10 / 0.1), 0 0 0 1px oklch(0.55 0.22 10 / 0.1)",
  fontSize: "13px",
  fontFamily: "'JetBrains Mono', monospace",
  padding: "10px 14px",
}

export function JobTrendChart({
  data,
  title,
  description,
}: {
  data: JobTrendPoint[]
  title: string
  description: string
}) {
  return (
    <Card className="group overflow-hidden border-border bg-card transition-all duration-300 hover:infrared-glow">
      <CardHeader>
        <CardTitle>{title}</CardTitle>
        <CardDescription>{description}</CardDescription>
      </CardHeader>
      <CardContent>
        {data.length ? (
          <div className="h-[220px]">
            <ResponsiveContainer>
              <AreaChart data={data} margin={{ left: 0, right: 8, top: 8, bottom: 0 }}>
                <defs>
                  <linearGradient id="trendGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="var(--primary)" stopOpacity={0.35} />
                    <stop offset="100%" stopColor="var(--primary)" stopOpacity={0.04} />
                  </linearGradient>
                </defs>
                <CartesianGrid
                  stroke="var(--border)"
                  strokeDasharray="4 4"
                  vertical={false}
                />
                <XAxis
                  dataKey="label"
                  axisLine={false}
                  tickLine={false}
                  tickMargin={12}
                  tick={{ fontSize: 12, fill: "var(--muted-foreground)" }}
                />
                <YAxis
                  axisLine={false}
                  tickLine={false}
                  tickMargin={12}
                  width={48}
                  domain={[0, 100]}
                  tickFormatter={(v) => Number(v).toFixed(0)}
                  tick={{ fontSize: 12, fill: "var(--muted-foreground)", fontFamily: "'JetBrains Mono', monospace" }}
                />
                <Tooltip
                  cursor={{ stroke: "var(--border)", strokeDasharray: "4 4" }}
                  contentStyle={tooltipStyle}
                  formatter={(v) => [Number(v).toFixed(2), "平均分"]}
                />
                <Area
                  type="monotone"
                  dataKey="score"
                  stroke="var(--primary)"
                  strokeWidth={2.5}
                  fill="url(#trendGrad)"
                  dot={false}
                  activeDot={{ r: 5, fill: "var(--primary)", stroke: "var(--card)", strokeWidth: 2 }}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <EmptyChart description="暂无趋势数据" />
        )}
      </CardContent>
    </Card>
  )
}

export function ScoreDistributionChart({
  data,
  title,
  description,
}: {
  data: ScoreBandPoint[]
  title: string
  description: string
}) {
  return (
    <Card className="group overflow-hidden border-border bg-card transition-all duration-300 hover:infrared-glow">
      <CardHeader>
        <CardTitle>{title}</CardTitle>
        <CardDescription>{description}</CardDescription>
      </CardHeader>
      <CardContent>
        {data.some((d) => d.count > 0) ? (
          <div className="h-[220px]">
            <ResponsiveContainer>
              <BarChart data={data} margin={{ left: 0, right: 8, top: 8, bottom: 0 }}>
                <CartesianGrid
                  stroke="var(--border)"
                  strokeDasharray="4 4"
                  vertical={false}
                />
                <XAxis
                  dataKey="name"
                  axisLine={false}
                  tickLine={false}
                  tickMargin={12}
                  tick={{ fontSize: 12, fill: "var(--muted-foreground)" }}
                />
                <YAxis
                  allowDecimals={false}
                  axisLine={false}
                  tickLine={false}
                  tickMargin={12}
                  width={36}
                  tick={{ fontSize: 12, fill: "var(--muted-foreground)", fontFamily: "'JetBrains Mono', monospace" }}
                />
                <Tooltip
                  cursor={{ fill: "color-mix(in oklab, var(--primary) 8%, transparent)" }}
                  contentStyle={tooltipStyle}
                  formatter={(v) => [Number(v), "图片数"]}
                />
                <Bar
                  dataKey="count"
                  fill="var(--primary)"
                  radius={[8, 8, 0, 0]}
                  opacity={0.85}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <EmptyChart description="暂无分布数据" />
        )}
      </CardContent>
    </Card>
  )
}

export function RankedScoreChart({
  data,
  title,
  description,
}: {
  data: RankedScorePoint[]
  title: string
  description: string
}) {
  return (
    <Card className="group overflow-hidden border-border bg-card transition-all duration-300 hover:infrared-glow">
      <CardHeader>
        <CardTitle>{title}</CardTitle>
        <CardDescription>{description}</CardDescription>
      </CardHeader>
      <CardContent>
        {data.length ? (
          <div className="h-[220px]">
            <ResponsiveContainer>
              <LineChart data={data} margin={{ left: 0, right: 8, top: 8, bottom: 0 }}>
                <CartesianGrid
                  stroke="var(--border)"
                  strokeDasharray="4 4"
                  vertical={false}
                />
                <XAxis
                  dataKey="rank"
                  axisLine={false}
                  tickLine={false}
                  tickMargin={12}
                  tick={{ fontSize: 12, fill: "var(--muted-foreground)", fontFamily: "'JetBrains Mono', monospace" }}
                />
                <YAxis
                  axisLine={false}
                  tickLine={false}
                  tickMargin={12}
                  width={48}
                  domain={[0, 100]}
                  tickFormatter={(v) => Number(v).toFixed(0)}
                  tick={{ fontSize: 12, fill: "var(--muted-foreground)", fontFamily: "'JetBrains Mono', monospace" }}
                />
                <Tooltip
                  cursor={{ stroke: "var(--border)", strokeDasharray: "4 4" }}
                  contentStyle={tooltipStyle}
                  formatter={(v) => [Number(v).toFixed(2), "质量分"]}
                />
                <Line
                  type="monotone"
                  dataKey="score"
                  stroke="var(--primary)"
                  strokeWidth={2.5}
                  dot={false}
                  activeDot={{
                    r: 5,
                    fill: "var(--primary)",
                    stroke: "var(--card)",
                    strokeWidth: 2,
                  }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <EmptyChart description="暂无排序数据" />
        )}
      </CardContent>
    </Card>
  )
}
