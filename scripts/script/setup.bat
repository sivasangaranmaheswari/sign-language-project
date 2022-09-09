@echo off
cd scripts
call welcome
@ECHO OFF
:BEGIN
CLS
echo Sign Language Setup
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
echo Setting up Number Classifier...
call number-setup
pause
GOTO BEGIN
:ONE
echo Setting up Alphabet Classifier...
call alphabet-setup
pause
GOTO BEGIN
:END
pause