@echo off
set num = 0
For /r  . %%i in (*.jpeg) do (
set /a num += 1
echo %%i
ren %%i *.jpg) 
echo 共%num%个文件被处理成功
pause>nul
