@echo off
echo Starting...
call %ANACONDA_PATH%\activate tensorflow
call python "..\source\classify-numbers.py" -model_path "..\models\numbers\number-classifier.h5" -temp "..\temp" -mail yes
call deactivate
