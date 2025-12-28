@echo off
echo 🚀 Deploying Video Game Sales Dashboard to Streamlit Cloud
echo.

cd /d "C:\Users\akank\OneDrive\Desktop\learn"

echo 📋 Checking if repository is up to date...
"C:\Program Files\Git\cmd\git.exe" status

echo.
echo 📦 Committing any changes...
"C:\Program Files\Git\cmd\git.exe" add .
"C:\Program Files\Git\cmd\git.exe" commit -m "Prepare for Streamlit Cloud deployment" || echo "No changes to commit"

echo.
echo 📤 Pushing to GitHub...
"C:\Program Files\Git\cmd\git.exe" push origin master

echo.
echo 🎉 Ready for Streamlit Cloud deployment!
echo.
echo 🌐 Go to: https://share.streamlit.io
echo 📋 Deployment settings:
echo    • Repository: bodhankarakanksha562/video-game-sales-dashboard
echo    • Branch: master
echo    • Main file: app.py
echo    • Python version: 3.9
echo.
echo 🚀 Click 'Deploy!' and your app will be live!
echo.
pause