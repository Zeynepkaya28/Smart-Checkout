# Smart Checkout – Windows .exe derleme betiği
# Kullanım (proje klasöründe, venv ETKİN ve requirements kurulu iken):
#     .\build_exe.ps1
#     .\build_exe.ps1 -WithModel      # yolov8m-oiv7.pt modelini de pakete koyar (internetsiz ilk açılış)
param(
    [switch]$WithModel,
    [string]$ModelSize = "m"
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not $env:VIRTUAL_ENV) {
    Write-Warning "Sanal ortam etkin değil. Önce: .\.venv\Scripts\Activate.ps1"
}

Write-Host "1/4 PyTorch / CUDA kontrolü..." -ForegroundColor Cyan
python -c "import torch; print('   CUDA:', torch.cuda.is_available(), '| torch', torch.__version__)"
if ($LASTEXITCODE -ne 0) { throw "PyTorch kurulu değil. README'deki kurulum adımlarını izleyin." }

Write-Host "2/4 PyInstaller kuruluyor..." -ForegroundColor Cyan
python -m pip install --quiet --upgrade "pyinstaller>=6.10" "pyinstaller-hooks-contrib>=2024.8"

Write-Host "3/4 Paketleniyor (birkaç dakika sürebilir)..." -ForegroundColor Cyan
python -m PyInstaller --noconfirm --clean smart_checkout.spec
if ($LASTEXITCODE -ne 0) { throw "PyInstaller derlemesi başarısız." }

$dist = Join-Path $PSScriptRoot "dist\SmartCheckout"
Write-Host "4/4 Ek dosyalar kopyalanıyor..." -ForegroundColor Cyan
Copy-Item README.md $dist -Force
if ($WithModel) {
    $model = "yolov8$ModelSize-oiv7.pt"
    if (-not (Test-Path $model)) {
        Write-Host "   $model indiriliyor..."
        python -c "from ultralytics import YOLO; YOLO('$model')"
    }
    Copy-Item $model $dist -Force
    Write-Host "   $model pakete eklendi."
}

$sizeGb = [math]::Round(((Get-ChildItem $dist -Recurse | Measure-Object Length -Sum).Sum / 1GB), 2)
Write-Host ""
Write-Host "Tamamlandı: $dist\SmartCheckout.exe  ($sizeGb GB)" -ForegroundColor Green
Write-Host "Dağıtırken dist\SmartCheckout klasörünün TAMAMINI kopyalayın (ör. zip'leyerek)."
