@echo off
chcp 65001 >nul
rem ===== 体内置换对照：生成置换训练集 P，再训练三个种子的 P 网络（各 120 轮，约 4-5 小时） =====
rem 先核对下面 4 行路径（可从 run_confirm.bat 复制）
set PY=python
set H_DIR=G:\2026.3\代码\paper_results\geology_v4_20260922\dataset_11600
set CONFIG=G:\2026.3\代码\paper_results\geology_v4_20260922\train10000_seed20260727\metadata.json
set G_FILE=请改成G.mat的完整路径
set P_DIR=%~dp0dataset_11600_permuted
set OUT=%~dp0runs_permuted

%PY% -c "import torch,sys; ok=torch.cuda.is_available(); print('torch',torch.__version__,'CUDA',ok, torch.cuda.get_device_name(0) if ok else ''); sys.exit(0 if ok else 1)"
if errorlevel 1 ( echo 当前 Python 没有 CUDA，请先激活原来训练用的环境 & pause & exit /b 1 )

if not exist "%P_DIR%\DONE" (
  echo 正在生成置换训练集（约几分钟）...
  %PY% -X utf8 "%~dp0make_permuted_library.py" --h-dir "%H_DIR%" --G "%G_FILE%" --out "%P_DIR%"
  if errorlevel 1 ( echo 生成置换训练集出错，已停止 & pause & exit /b 1 )
)

call :RUN 20260727 || goto :ERR
call :RUN 20260728 || goto :ERR
call :RUN 20260729 || goto :ERR
echo 全部完成
pause
exit /b 0

:RUN
if exist "%OUT%\P_seed%1_lg0\DONE" exit /b 0
set RES=
if exist "%OUT%\P_seed%1_lg0\last_checkpoint.pth" set RES=--resume
%PY% -X utf8 "%~dp0train_v2.py" --data-dir "%P_DIR%" --config-json "%CONFIG%" --G "%G_FILE%" --seed %1 --arm P --lambda-g 0 --epochs 120 --out "%OUT%" %RES%
exit /b %errorlevel%

:ERR
echo 出错，已停止（再次双击会从断点继续）
pause
exit /b 1
