@echo off
echo 🚀 Pushing Video Game Sales Dashboard to GitHub
echo.

cd /d "C:\Users\akank\OneDrive\Desktop\learn"

set GH_EXE="C:\Program Files\GitHub CLI\gh.exe"

echo 📋 Checking GitHub authentication...
%GH_EXE% auth status
if %errorlevel% neq 0 (
    echo ❌ Please run 'gh auth login' first to authenticate with GitHub
    echo.
    echo To authenticate, run this command:
    echo %GH_EXE% auth login
    echo.
    pause
    exit /b 1
)

echo.
echo 📦 Creating GitHub repository and pushing code...
%GH_EXE% repo create video-game-sales-dashboard --public --source=. --remote=origin --push

if %errorlevel% equ 0 (
    echo.
    echo ✅ Success! Your code has been pushed to GitHub
    echo.
    echo 🌐 Find your repository at:
    %GH_EXE% repo view video-game-sales-dashboard --web
    echo.
    echo 📝 Note: Replace YOUR_USERNAME in the URL with your actual GitHub username
    echo    Example: https://github.com/johndoe/video-game-sales-dashboard
    echo.
    echo 🎉 Your Video Game Sales Dashboard is now live on GitHub!
) else (
    echo.
    echo ❌ Failed to create repository. Please check the error messages above.
    echo.
    echo 💡 Troubleshooting:
    echo 1. Make sure you're authenticated: %GH_EXE% auth login
    echo 2. Check if repository name already exists
    echo 3. Try a different repository name
)

echo.
pause

echo.
pause