@echo off
set /p filename=Enter model file name: 
if "%filename%"=="" goto error
call %ANACONDA_PATH%\activate tensorflow
call python "..\source\setup-alphabets.py" -dataset_path "../dataset/alphabets" -model_path "../models/alphabets" -model_name %filename% -epochs 300 -patience 30 -out_file "../models/model-summary/alphabets/summary.txt" -min_lr 0.001
echo Model Built Successfully.  
pause
call deactivate
goto end
:error
echo You must enter a file name to continue
:end

