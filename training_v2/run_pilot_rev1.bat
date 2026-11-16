@echo off
chcp 65001 >nul
rem ===== 正演损失项预试验：H 库、种子 20260727、修订1 追加 lambda_g 0.003 与 0.01、各 40 轮（约 1.6 小时） =====
rem 先核对下面 4 行路径（原训练脚本里 ROOT = G:\2026.3\代码\paper_results\geology_v4_20260922）
set PY=python
set H_DIR=G:\2026.3\代码\paper_results\geology_v4_20260922\dataset_11600
set CONFIG=G:\2026.3\代码\paper_results\geology_v4_20260922\train10000_seed20260727\metadata.json
set G_FILE=请改成G.mat的完整路径
set OUT=%~dp0runs_pilot

%PY% -c "import torch,sys; ok=torch.cuda.is_available(); print('torch',torch.__version__,'CUDA',ok, torch.cuda.get_device_name(0) if ok else ''); sys.exit(0 if ok else 1)"
if errorlevel 1 ( echo 当前 Python 没有 CUDA，请先激活原来训练用的环境 & pause & exit /b 1 )

for %%L in (0.003 0.01) do (
  if not exist "%OUT%\H_seed20260727_lg%%L\DONE" (
    if exist "%OUT%\H_seed20260727_lg%%L\last_checkpoint.pth" (
      %PY% -X utf8 "%~dp0train_v2.py" --data-dir "%H_DIR%" --config-json "%CONFIG%" --G "%G_FILE%" --seed 20260727 --arm H --lambda-g %%L --epochs 40 --snapshots 40 --out "%OUT%" --resume
    ) else (
      %PY% -X utf8 "%~dp0train_v2.py" --data-dir "%H_DIR%" --config-json "%CONFIG%" --G "%G_FILE%" --seed 20260727 --arm H --lambda-g %%L --epochs 40 --snapshots 40 --out "%OUT%"
    )
    if errorlevel 1 ( echo lambda_g=%%L 出错，已停止 & pause & exit /b 1 )
  )
)
%PY% -X utf8 "%~dp0summarize_pilot.py" "%OUT%"
pause
