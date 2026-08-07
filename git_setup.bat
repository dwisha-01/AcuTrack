@echo off
echo ====================================================
echo             AcuTrack GitHub Upload Setup
echo ====================================================
echo.
echo Choose an option:
echo [1] Start completely fresh (erase history, initialize new repo)
echo [2] Keep history (just change remote URL of existing repo)
echo.
set /p choice="Enter choice (1 or 2): "

if "%choice%"=="1" (
    echo.
    echo Resetting Git repository...
    rd /s /q .git
    git init
    git add .
    git commit -m "Initial commit"
    echo Git re-initialized cleanly!
)

echo.
set /p repo_url="Paste your new GitHub Repository URL (e.g., https://github.com/user/repo.git): "

git remote remove origin >nul 2>&1
git remote add origin %repo_url%
git branch -M main
echo.
echo Remote destination set to %repo_url%
echo.
echo Pushing code to GitHub...
git push -u origin main
echo.
echo ====================================================
echo Process complete!
echo ====================================================
pause
