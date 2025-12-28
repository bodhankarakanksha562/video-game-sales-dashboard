# PowerShell script to push Video Game Sales Dashboard to GitHub

Write-Host "🚀 Pushing Video Game Sales Dashboard to GitHub" -ForegroundColor Cyan
Write-Host ""

Set-Location "C:\Users\akank\OneDrive\Desktop\learn"

$ghPath = "C:\Program Files\GitHub CLI\gh.exe"

Write-Host "📋 Checking GitHub authentication..." -ForegroundColor Yellow

# Check if GitHub CLI is authenticated
& $ghPath auth status
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Please run 'gh auth login' first to authenticate with GitHub" -ForegroundColor Red
    Write-Host ""
    Write-Host "To authenticate, run:" -ForegroundColor Yellow
    Write-Host "& '$ghPath' auth login" -ForegroundColor White
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "📦 Creating GitHub repository and pushing code..." -ForegroundColor Yellow

# Create repository and push
& $ghPath repo create video-game-sales-dashboard --public --source=. --remote=origin --push

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "✅ Success! Your code has been pushed to GitHub" -ForegroundColor Green
    Write-Host ""
    Write-Host "🌐 Find your repository at:" -ForegroundColor Cyan
    & $ghPath repo view video-game-sales-dashboard --web
    Write-Host ""
    Write-Host "📝 Note: Replace YOUR_USERNAME in the URL with your actual GitHub username" -ForegroundColor Gray
    Write-Host "   Example: https://github.com/johndoe/video-game-sales-dashboard" -ForegroundColor Gray
    Write-Host ""
    Write-Host "🎉 Your Video Game Sales Dashboard is now live on GitHub!" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "❌ Failed to create repository. Please check the error messages above." -ForegroundColor Red
    Write-Host ""
    Write-Host "💡 Troubleshooting:" -ForegroundColor Yellow
    Write-Host "1. Make sure you're authenticated: & '$ghPath' auth login" -ForegroundColor White
    Write-Host "2. Check if repository name already exists" -ForegroundColor White
    Write-Host "3. Try a different repository name" -ForegroundColor White
}

Write-Host ""
Read-Host "Press Enter to exit"

Write-Host ""
Read-Host "Press Enter to exit"