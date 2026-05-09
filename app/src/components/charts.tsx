import { Area, AreaChart, Bar, BarChart, CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts"
import type { JobTrendPoint, RankedScorePoint, ScoreBandPoint } from "@/lib/analytics"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"

function EmptyChart({ description }: { description: string }) {
  return (
    <div className="flex h-[220px] items-center justify-center rounded-xl border border-dashed bg-muted/20 px-6 text-center text-sm text-muted-foreground">
      {description}
    </div>
  )
}

const ts = { borderRadius: 12, borderColor: "var(--border)", background: "color-mix(in oklab, var(--card) 92%, transparent)" }

export function JobTrendChart({ data, title, description }: { data: JobTrendPoint[]; title: string; description: string }) {
  return (
    <Card className="overflow-hidden">
      <CardHeader><CardTitle>{title}</CardTitle><CardDescription>{description}</CardDescription></CardHeader>
      <CardContent>
        {data.length ? (
          <div className="h-[220px]"><ResponsiveContainer>
            <AreaChart data={data} margin={{ left: 0, right: 8, top: 8, bottom: 0 }}>
              <defs><linearGradient id="t" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="var(--primary)" stopOpacity={0.35} /><stop offset="100%" stopColor="var(--primary)" stopOpacity={0.04} /></linearGradient></defs>
              <CartesianGrid stroke="var(--border)" strokeDasharray="4 4" vertical={false} />
              <XAxis dataKey="label" axisLine={false} tickLine={false} tickMargin={12} />
              <YAxis axisLine={false} tickLine={false} tickMargin={12} width={42} domain={[0, 1]} tickFormatter={(v) => Number(v).toFixed(1)} />
              <Tooltip cursor={{ stroke: "var(--border)", strokeDasharray: "4 4" }} contentStyle={ts} formatter={(v) => [Number(v).toFixed(4), "平均分"]} />
              <Area type="monotone" dataKey="score" stroke="var(--primary)" strokeWidth={2} fill="url(#t)" />
            </AreaChart>
          </ResponsiveContainer></div>
        ) : <EmptyChart description="暂无趋势数据" />}
      </CardContent>
    </Card>
  )
}

export function ScoreDistributionChart({ data, title, description }: { data: ScoreBandPoint[]; title: string; description: string }) {
  return (
    <Card className="overflow-hidden">
      <CardHeader><CardTitle>{title}</CardTitle><CardDescription>{description}</CardDescription></CardHeader>
      <CardContent>
        {data.some((d) => d.count > 0) ? (
          <div className="h-[220px]"><ResponsiveContainer>
            <BarChart data={data} margin={{ left: 0, right: 8, top: 8, bottom: 0 }}>
              <CartesianGrid stroke="var(--border)" strokeDasharray="4 4" vertical={false} />
              <XAxis dataKey="name" axisLine={false} tickLine={false} tickMargin={12} />
              <YAxis allowDecimals={false} axisLine={false} tickLine={false} tickMargin={12} width={36} />
              <Tooltip cursor={{ fill: "color-mix(in oklab, var(--primary) 8%, transparent)" }} contentStyle={ts} formatter={(v) => [Number(v), "图片数"]} />
              <Bar dataKey="count" fill="color-mix(in oklab, var(--primary) 70%, white)" radius={[8, 8, 0, 0]} />
            </BarChart>
          </ResponsiveContainer></div>
        ) : <EmptyChart description="暂无分布数据" />}
      </CardContent>
    </Card>
  )
}

export function RankedScoreChart({ data, title, description }: { data: RankedScorePoint[]; title: string; description: string }) {
  return (
    <Card className="overflow-hidden">
      <CardHeader><CardTitle>{title}</CardTitle><CardDescription>{description}</CardDescription></CardHeader>
      <CardContent>
        {data.length ? (
          <div className="h-[220px]"><ResponsiveContainer>
            <LineChart data={data} margin={{ left: 0, right: 8, top: 8, bottom: 0 }}>
              <CartesianGrid stroke="var(--border)" strokeDasharray="4 4" vertical={false} />
              <XAxis dataKey="rank" axisLine={false} tickLine={false} tickMargin={12} />
              <YAxis axisLine={false} tickLine={false} tickMargin={12} width={42} domain={[0, 1]} tickFormatter={(v) => Number(v).toFixed(1)} />
              <Tooltip cursor={{ stroke: "var(--border)", strokeDasharray: "4 4" }} contentStyle={ts} formatter={(v) => [Number(v).toFixed(4), "质量分"]} />
              <Line type="monotone" dataKey="score" stroke="color-mix(in oklab, var(--primary) 80%, white)" strokeWidth={2.5} dot={false} activeDot={{ r: 5 }} />
            </LineChart>
          </ResponsiveContainer></div>
        ) : <EmptyChart description="暂无排序数据" />}
      </CardContent>
    </Card>
  )
}
