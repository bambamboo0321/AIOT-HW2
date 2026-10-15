"""以專案 .venv Kernel 從頭執行 Phase 2 Notebook，驗收與 CLI 一致。"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

# Kernel 啟動前將可寫入的設定／快取限定在本專案。
for name, relative in {
    "JUPYTER_CONFIG_DIR": ".venv/jupyter-config",
    "JUPYTER_DATA_DIR": ".venv/share/jupyter",
    "JUPYTER_RUNTIME_DIR": ".venv/jupyter-runtime",
    "IPYTHONDIR": ".venv/ipython",
    "MPLCONFIGDIR": ".venv/matplotlib-config",
}.items():
    os.environ[name] = str(ROOT / relative)

import nbformat
from nbclient import NotebookClient
from jupyter_client import AsyncKernelManager
from jupyter_client.kernelspec import KernelSpecManager

from src.phase2_common import CSV_PATH, REPORT_DIR, write_json
from src.validate_data import sha256_file


class ProjectKernelManager(AsyncKernelManager):
    def __init__(self, **kwargs):
        kwargs["kernel_spec_manager"] = KernelSpecManager(
            kernel_dirs=[str(ROOT / ".venv/share/jupyter/kernels")]
        )
        super().__init__(**kwargs)


def main() -> int:
    started_at = datetime.now(timezone.utc).isoformat()
    execution_path = REPORT_DIR / "notebook_execution.json"
    previous = json.loads(execution_path.read_text()) if execution_path.is_file() else None
    protected_paths = [CSV_PATH, ROOT / "KAG_energydata_complete.csv",
                       ROOT / "reports/phase1/data_validation.json",
                       REPORT_DIR / "baseline_metrics.json", REPORT_DIR / "model_evaluation.md",
                       REPORT_DIR / "split_indices.json",
                       *sorted((REPORT_DIR / "models").glob("*")),
                       *sorted(REPORT_DIR.glob("predictions_*.csv"))]
    protected_hashes = {path: sha256_file(path) for path in protected_paths if path.is_file()}
    kernel_dir = ROOT / ".venv/share/jupyter/kernels/aiot-hw2"
    if not (kernel_dir / "kernel.json").is_file():
        result = subprocess.run(
            [str(ROOT / ".venv/bin/python"), "-m", "ipykernel", "install", "--prefix",
             str(ROOT / ".venv"), "--name", "aiot-hw2", "--display-name", "AIOT-HW2 (.venv)"],
            capture_output=True, text=True,
        )
        if result.returncode:
            raise RuntimeError("無法在 .venv 建立專案 Kernel。")
    (ROOT / ".venv/jupyter-runtime").mkdir(parents=True, exist_ok=True)
    notebook_path = ROOT / "notebooks/phase2_exploration.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    references = {}
    for name in ("eda_statistics.json", "baseline_metrics.json"):
        if not (REPORT_DIR / name).is_file():
            raise RuntimeError(f"缺少既有參考報告：{name}；本同步流程不重訓 Baseline。")
        references[name] = json.loads((REPORT_DIR / name).read_text())
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    client = NotebookClient(
        notebook, kernel_name="aiot-hw2", kernel_manager_class=ProjectKernelManager,
        timeout=180, resources={"metadata": {"path": str(ROOT / "notebooks")}},
    )
    client.execute()
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    if any(cell.execution_count is None for cell in code_cells):
        raise RuntimeError("Notebook 存在未執行的 Code Cell。")
    if any(output.output_type == "error" for cell in code_cells for output in cell.outputs):
        raise RuntimeError("Notebook 存在錯誤輸出。")
    for name, expected in references.items():
        if json.loads((REPORT_DIR / name).read_text()) != expected:
            raise RuntimeError(f"Notebook 與既有 CLI 結果不一致：{name}；停止並檢查差異。")
    for path, expected in protected_hashes.items():
        if sha256_file(path) != expected:
            raise RuntimeError(f"受保護檔案在 Notebook 執行期間改變：{path.relative_to(ROOT)}")
    result = {
        "status": "PASS", "notebook": "notebooks/phase2_exploration.ipynb",
        "purpose": "phase2_notebook_sync", "started_at_utc": started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution_mode": "fresh kernel; cleared outputs; Run All",
        "code_cells_executed": len(code_cells), "error_outputs": 0,
        "kernel": "aiot-hw2", "kernel_environment": ".venv",
        "cli_eda_and_baseline_json_equal": len(references) == 2,
        "reference_reports_compared": list(references),
        "eda_figure_count": len(references["eda_statistics.json"]["figures"]),
        "evaluation_figure_count": len(references["baseline_metrics.json"]["figures"]),
        "baseline_execution": "load saved models; verify predictions and metrics; no fit or model writes",
        "protected_file_sha256": {path.relative_to(ROOT).as_posix(): checksum
                                  for path, checksum in protected_hashes.items()},
        "protected_files_unchanged": True,
        "previous_execution": previous,
    }
    notebook.metadata["phase2_execution"] = result
    nbformat.validate(notebook)
    nbformat.write(notebook, notebook_path)
    write_json(REPORT_DIR / "notebook_execution.json", result)
    print(f"PASS: Notebook 全部 {len(code_cells)} 個 Code Cells 成功；Kernel 已關閉。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
