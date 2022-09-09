@echo off
call welcome
@ECHO OFF
:BEGIN
CLS
echo Sign Language Recognition Software
echo Made By:
echo ---------------------------------------------------------------------------------------------------------
echo Sivasangaran V               110117083
echo Shally Preethika Mani        110117077
echo ---------------------------------------------------------------------------------------------------------
echo [1] Alphabets
echo [2] Numbers
echo [3] Quit
CHOICE /N /C:123
IF ERRORLEVEL ==3 GOTO THREE
IF ERRORLEVEL ==2 GOTO TWO
IF ERRORLEVEL ==1 GOTO ONE
GOTO END
:THREE
ECHO Quitting...
GOTO END
:TWO
echo Starting Number Classifier...
call number-classifier
pause
GOTO BEGIN
:ONE
echo Starting Alphabet Classifier...
call alphabet-classifier
pause
GOTO BEGIN
:END
pause