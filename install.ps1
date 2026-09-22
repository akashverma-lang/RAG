# Local RAG - one-line install for Windows.
#
#   irm https://raw.githubusercontent.com/akashverma-lang/RAG/main/install.ps1 | iex
#
# Downloads the app, sets up a private Python environment and starts it. Everything
# lands in one folder that can be deleted to uninstall.

$ErrorActionPreference = "Stop"

# ---- edit these two lines to your repository ------------------------------------
$Owner  = if ($env:RAG_OWNER) { $env:RAG_OWNER } else { "akashverma-lang" }
$Repo   = if ($env:RAG_REPO)  { $env:RAG_REPO }  else { "RAG" }
$Branch = if ($env:RAG_BRANCH){ $env:RAG_BRANCH } else { "main" }
# ---------------------------------------------------------------------------------

$Target = if ($env:RAG_HOME) { $env:RAG_HOME } else { Join-Path $HOME "LocalRAG" }

function Say($msg)  { Write-Host "  $msg" }
function Step($msg) { Write-Host "`n>> $msg" -ForegroundColor Cyan }

Write-Host ""
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host "  Local RAG - installing" -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan

# --- Python -----------------------------------------------------------------------
Step "Looking for Python"
$py = $null
foreach ($candidate in @("python", "python3", "py")) {
    try {
        $v = & $candidate --version 2>&1
        if ($v -match "Python (\d+)\.(\d+)") {
            if ([int]$Matches[1] -ge 3 -and [int]$Matches[2] -ge 10) { $py = $candidate; break }
            Say "found $v, which is too old"
        }
    } catch { }
}
if (-not $py) {
    Write-Host ""
    Write-Host "  Python 3.10 or newer is required." -ForegroundColor Yellow
    Write-Host "  Install it from https://www.python.org/downloads/"
    Write-Host "  and tick 'Add python.exe to PATH' during installation."
    Write-Host "  Then run this command again."
    return
}
Say "using $(& $py --version)"

# --- the code ---------------------------------------------------------------------
Step "Downloading Local RAG"
$zip = Join-Path $env:TEMP "localrag.zip"
$url = "https://github.com/$Owner/$Repo/archive/refs/heads/$Branch.zip"
try {
    Invoke-WebRequest -Uri $url -OutFile $zip -UseBasicParsing
} catch {
    Write-Host "  Could not download $url" -ForegroundColor Red
    Write-Host "  Check the repository name, or that it is public."
    return
}

# Keep whatever settings and index a previous install left behind.
$stage = Join-Path $env:TEMP "localrag-unpack"
if (Test-Path $stage) { Remove-Item $stage -Recurse -Force }
Expand-Archive -Path $zip -DestinationPath $stage -Force
$unpacked = Get-ChildItem $stage | Select-Object -First 1

New-Item -ItemType Directory -Force -Path $Target | Out-Null
Copy-Item -Path (Join-Path $unpacked.FullName "*") -Destination $Target -Recurse -Force
Remove-Item $zip, $stage -Recurse -Force -ErrorAction SilentlyContinue
Say "installed to $Target"

# --- environment ------------------------------------------------------------------
Step "Setting up a private Python environment"
Push-Location $Target
try {
    if (-not (Test-Path ".venv\Scripts\python.exe")) { & $py -m venv .venv }
    $venvPy = Join-Path $Target ".venv\Scripts\python.exe"

    Say "installing packages - this takes a few minutes the first time"
    & $venvPy -m pip install --upgrade pip --quiet
    & $venvPy -m pip install -r requirements.txt --quiet
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  Package installation failed." -ForegroundColor Red
        return
    }

    # A shortcut, so it can be started again without retyping any of this.
    $lnk = Join-Path ([Environment]::GetFolderPath("Desktop")) "Local RAG.lnk"
    $shell = New-Object -ComObject WScript.Shell
    $s = $shell.CreateShortcut($lnk)
    $s.TargetPath = $venvPy
    $s.Arguments = "launcher.py"
    $s.WorkingDirectory = $Target
    $s.Description = "Ask questions about your own documents"
    $s.Save()
    Say "desktop shortcut created"

    Write-Host ""
    Write-Host "==============================================" -ForegroundColor Green
    Write-Host "  Installed. Starting it now." -ForegroundColor Green
    Write-Host "==============================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "  A setup screen will ask for your documents folder"
    Write-Host "  and a free API key:"
    Write-Host "     Groq    console.groq.com/keys"
    Write-Host "     Gemini  aistudio.google.com/apikey"
    Write-Host ""
    Write-Host "  Start it again later from the desktop shortcut,"
    Write-Host "  or by running run.bat in $Target"
    Write-Host ""

    & $venvPy launcher.py
} finally {
    Pop-Location
}
