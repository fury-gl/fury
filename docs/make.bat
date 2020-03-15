@ECHO OFF

pushd %~dp0

REM Command file for Sphinx documentation

if "%SPHINXBUILD%" == "" (
	set SPHINXBUILD=sphinx-build
)
set SOURCEDIR=source
set BUILDDIR=build
set SPHINXPROJ=FURY

if "%1" == "" goto help

%SPHINXBUILD% >NUL 2>NUL
if errorlevel 9009 (
	echo.
	echo.The 'sphinx-build' command was not found. Make sure you have Sphinx
	echo.installed, then set the SPHINXBUILD environment variable to point
	echo.to the full path of the 'sphinx-build' executable. Alternatively you
	echo.may add the Sphinx directory to PATH.
	echo.
	echo.If you don't have Sphinx installed, grab it from
	echo.http://sphinx-doc.org/
	exit /b 1
)

if "%1" == "help" (
	:help
	%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR% %SPHINXOPTS%
	goto end
)

if "%1" == "clean" (
	:clean
	del /q /s %SOURCEDIR%\\api %SOURCEDIR%\\auto_examples %SOURCEDIR%\\auto_tutorials %SOURCEDIR%\\reference
	rmdir %SOURCEDIR%\\api %SOURCEDIR%\\auto_examples %SOURCEDIR%\\auto_tutorials %SOURCEDIR%\\reference
	%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% clean
	exit /B
)

if "%1" == "upload" (
	:upload
	python upload_to_gh-pages.py
	exit /B
)


%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS%
goto end

:end
popd
