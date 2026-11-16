@echo off
chcp 65001 >nul
rem ===== 正演损失项确认训练：种子 20260727，C/H 各 lambda_g 0.003 与 0，各 120 轮（4 个网络，约 6 小时） =====
rem 先核对下面 5 行路径
set PY=python
set H_DIR=G:\2026.3\代码\paper_results\geology_v4_20260922\dataset_11600
set C_DIR=G:\2026.3\代码\C盘工作区_20261020\manuscript_revision_20261015\constant_reference_11600
set CONFIG=G:\2026.3\代码\paper_results\geology_v4_20260922\train10000_seed20260727\metadata.json
set G_FILE=请改成G.mat的完整路径
set OUT=%~dp0runs_confirm

%PY% -c "import torch,sys; ok=torch.cuda.is_available(); print('torch',torch.__version__,'CUDA',ok, torch.cuda.get_device_name(0) if ok else ''); sys.exit(0 if ok else 1)"
if errorlevel 1 ( echo 当前 Python 没有 CUDA，请先激活原来训练用的环境 & pause & exit /b 1 )

call :RUN H "%H_DIR%" 0.003 || goto :ERR
call :RUN C "%C_DIR%" 0.003 || goto :ERR
call :RUN C "%C_DIR%" 0 || goto :ERR
call :RUN H "%H_DIR%" 0 || goto :ERR
echo 全部完成
pause
exit /b 0

:RUN
if exist "%OUT%\%1_seed20260727_lg%3\DONE" exit /b 0
set RES=
if exist "%OUT%\%1_seed20260727_lg%3\last_checkpoint.pth" set RES=--resume
%PY% -X utf8 "%~dp0train_v2.py" --data-dir %2 --config-json "%CONFIG%" --G "%G_FILE%" --seed 20260727 --arm %1 --lambda-g %3 --epochs 120 --out "%OUT%" %RES%
exit /b %errorlevel%

:ERR
echo 出错，已停止（再次双击会从断点继续）
pause
exit /b 1
