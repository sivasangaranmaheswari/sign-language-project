@echo off
echo Starting...
call %ANACONDA_PATH%\activate tensorflow
call python "..\source\classify-alphabets.py" -model_path "..\models\alphabets\alpha-classifier-27-03-2021-13-02.h5" -temp "..\temp" -mail yes
call deactivate
