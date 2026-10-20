// Advisory UI capability only. M16 remains the authority for connection/security checks.
export type ServerComputeStatus = {available: boolean; message: string};
export function serverComputeStatus(environment: Record<string, string | undefined>, localPythonExecutable: boolean): ServerComputeStatus {
  const keys = ["AEP_BACKEND_URL", "AEP_BACKEND_TOKEN", "AEP_SESSION_SECRET", "AEP_PUBLIC_ORIGIN"];
  const present = keys.filter(key => environment[key] !== undefined).length;
  if (present === keys.length) return {available: true, message: "已提供 Server 連線設定；尚未確認服務健康狀態，實際可用性依運算回應。確認後 CSV 會傳送到該服務。"};
  if (present || environment.VERCEL) return {available: false, message: "Server Mode 目前未設定完整 Python 運算服務，無法執行。請手動選擇 Browser Mode 在瀏覽器運算；CSV 不會自動轉送。"};
  return localPythonExecutable
    ? {available: true, message: "使用本機 Python；尚未確認套件與運算狀態，實際可用性依運算回應。"}
    : {available: false, message: "Server Mode 找不到可執行的本機 Python，無法執行。請先設定 Python，或手動選擇 Browser Mode。"};
}
