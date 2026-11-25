@echo off
chcp 65001 >nul
rem ===== 全变密度训练集 A：把 H 中 20% 的均匀样本换成体内变化，再训练三个种子的 A 网络（各 120 轮，约 4-5 小时） =====
rem C 网络不用重训：A 的各异常体平均密度与 H 相同，其均匀对照就是现有的 C。
rem 先核对下面 4 行路径（与 run_permuted.bat 相同）
set PY=python
set H_DIR=G:\2026.3\代码\paper_results\geology_v4_20260922\dataset_11600
set CONFIG=G:\2026.3\代码\paper_results\geology_v4_20260922\train10000_seed20260727\metadata.json
set G_FILE=%USERPROFILE%\Desktop\G.mat
set A_DIR=%~dp0dataset_11600_allvar
set OUT=%~dp0runs_allvar

if not exist "%G_FILE%" ( echo 找不到 %G_FILE%，请修改 G_FILE 这一行 & pause & exit /b 1 )
if not exist "%H_DIR%\source_snapshot\geology_v4.py" ( echo 找不到 %H_DIR%\source_snapshot，请核对 H_DIR & pause & exit /b 1 )

%PY% -c "import torch,sys; ok=torch.cuda.is_available(); print('torch',torch.__version__,'CUDA',ok, torch.cuda.get_device_name(0) if ok else ''); sys.exit(0 if ok else 1)"
if errorlevel 1 ( echo 当前 Python 没有 CUDA，请先激活原来训练用的环境 & pause & exit /b 1 )

if not exist "%A_DIR%\DONE" (
  echo 正在生成全变密度训练集（约几分钟，会先逐例核对与原训练集一致）...
  %PY% -X utf8 "%~dp0make_allvar_library.py" --h-dir "%H_DIR%" --G "%G_FILE%" --out "%A_DIR%"
  if errorlevel 1 ( echo 生成训练集出错，已停止，请把窗口内容截图发给 Claude & pause & exit /b 1 )
)

call :RUN 20260727 || goto :ERR
call :RUN 20260728 || goto :ERR
call :RUN 20260729 || goto :ERR
echo 全部完成。请把 runs_allvar 文件夹和 dataset_11600_allvar\manifest.json 放到桌面
pause
exit /b 0

:RUN
if exist "%OUT%\A_seed%1_lg0\DONE" exit /b 0
set RES=
if exist "%OUT%\A_seed%1_lg0\last_checkpoint.pth" set RES=--resume
%PY% -X utf8 "%~dp0train_v2.py" --data-dir "%A_DIR%" --config-json "%CONFIG%" --G "%G_FILE%" --seed %1 --arm A --lambda-g 0 --epochs 120 --out "%OUT%" %RES%
exit /b %errorlevel%

:ERR
echo 出错，已停止（再次双击会从断点继续）
pause
exit /b 1
