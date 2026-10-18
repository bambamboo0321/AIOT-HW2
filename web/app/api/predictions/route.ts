import { loadPredictions } from "@/lib/load-predictions";
import { parseWindowQuery, predictionWindow } from "@/lib/prediction-window";
export const dynamic = "force-dynamic";
export async function GET(request: Request) {
  const snapshot = await loadPredictions();
  const headers = { "Cache-Control": "private, no-store" };
  if (snapshot.status !== "ready") return Response.json({ error: "已保存的預測資料尚未就緒，請重新同步。" }, { status: 503, headers });
  try {
    const query = parseWindowQuery(new URL(request.url).searchParams, snapshot.data.meta.totalRows);
    return Response.json(predictionWindow(snapshot.data.rows, snapshot.data.meta, query.start, query.size), { headers });
  } catch { return Response.json({ error: "視窗參數不合法；筆數僅支援 72、144、288。" }, { status: 400, headers }); }
}
