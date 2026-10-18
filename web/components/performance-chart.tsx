"use client";

import { useState } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

type Metric = "rmse" | "mae" | "r2";
type Point = { label: string; rmse: number; mae: number; r2: number };
export function PerformanceChart({ points, unit }: { points: Point[]; unit: string }) {
  const [metric, setMetric] = useState<Metric>("rmse");
  const rows = points.map((point, i) => ({ ...point, color: i === 0 ? "var(--baseline)" : "var(--selected)" }));
  const units = metric === "r2" ? "" : unit;
  return <div className="interactive-chart">
    <div className="chart-controls"><div className="chart-legend"><span><i className="legend-dot baseline"/>Baseline</span><span><i className="legend-dot selected"/>MI + LR</span></div>
      <div className="segmented-control" role="group" aria-label="圖表評估指標">{(["rmse", "mae", "r2"] as const).map(key => <button key={key} aria-pressed={metric === key} className={metric === key ? "selected" : ""} onClick={() => setMetric(key)}>{key === "r2" ? "R²" : key.toUpperCase()}</button>)}</div>
    </div>
    <div className="chart-heading"><span>{metric === "r2" ? "R² · higher is better" : `${metric.toUpperCase()} (${unit}) · lower is better`}</span><span>Fixed holdout</span></div>
    <div className="chart-canvas" role="img" aria-label={`固定 Holdout ${metric.toUpperCase()} 比較；完整數值見下方表格`}>
      <ResponsiveContainer width="100%" height="100%" minWidth={0} initialDimension={{ width: 500, height: 160 }}>
        <BarChart data={rows} layout="vertical" margin={{ top: 6, right: 25, bottom: 0, left: 0 }} accessibilityLayer>
          <CartesianGrid horizontal={false} stroke="var(--chart-grid)" strokeDasharray="3 4"/>
          <XAxis type="number" tick={{ fill: "var(--muted)", fontSize: 11 }} axisLine={false} tickLine={false} domain={[0, "auto"]}/>
          <YAxis type="category" dataKey="label" width={84} tick={{ fill: "var(--text-secondary)", fontSize: 12 }} axisLine={false} tickLine={false}/>
          <Tooltip cursor={{ fill: "var(--hover)" }} contentStyle={{ borderRadius: 10, background: "var(--surface)", border: "1px solid var(--border)", color: "var(--text)" }} formatter={(value) => [`${Number(value).toFixed(6)} ${units}`.trim(), metric === "r2" ? "R²" : metric.toUpperCase()]}/>
          <Bar dataKey={metric} name={metric.toUpperCase()} barSize={25} radius={[0, 5, 5, 0]} isAnimationActive={false} shape={(props) => {
            const { x, y, width, height, payload } = props;
            return <rect x={x} y={y} width={width} height={height} fill={payload.color} rx={4}/>;
          }}/>
        </BarChart>
      </ResponsiveContainer>
    </div>
  </div>;
}
