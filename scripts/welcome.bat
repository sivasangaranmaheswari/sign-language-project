@echo off
echo Loading...
call %ANACONDA_PATH%\activate tensorflow
call python "../source/welcome.py" -img "../glance/welcome.png"
rem pause
call deactivate
