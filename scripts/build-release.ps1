$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $PSScriptRoot
$Version = (Get-Content "$RepoRoot\VERSION").Trim()

Write-Host "=== pjev $Version release build (Windows x86_64) ===" -ForegroundColor Cyan

cmake -S $RepoRoot -B "$RepoRoot\build-release" `
    -DCMAKE_BUILD_TYPE=Release `
    -DBUILD_SHARED_LIBS=OFF

cmake --build "$RepoRoot\build-release" --config Release

Write-Host ""
Write-Host "=== Packaging ===" -ForegroundColor Cyan
Push-Location "$RepoRoot\build-release"
cpack -C Release
Pop-Location

Write-Host ""
Write-Host "=== Release artifacts ===" -ForegroundColor Cyan
Get-ChildItem "$RepoRoot\build-release\pjev-*.zip" -ErrorAction SilentlyContinue
