# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller paketleme tarifi – Smart Checkout (.exe)

Derleme:   pyinstaller --noconfirm --clean smart_checkout.spec
Çıktı:     dist/SmartCheckout/SmartCheckout.exe   (klasörün tamamı birlikte dağıtılmalı)

Tek dosya (onefile) yerine klasör (onedir) kullanılır: PyTorch + CUDA birkaç GB
tuttuğu için onefile her açılışta hepsini geçici klasöre açar ve çok yavaş başlar.
"""
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

ROOT = Path(SPECPATH)

datas = [
    (str(ROOT / "app.py"), "."),  # Streamlit app.py'yi dosyadan çalıştırır
    (str(ROOT / "products.json"), "."),  # ilk açılışta .exe'nin yanına kopyalanır
]
if (ROOT / "fonts").is_dir():
    datas.append((str(ROOT / "fonts"), "fonts"))
hiddenimports = ["streamlit.runtime.scriptrunner.magic_funcs"]


def safe(func, *args, **kwargs):
    """Paket kurulu değilse derlemeyi durdurmadan atla."""
    try:
        return func(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001
        print(f"[spec] atlandı: {func.__name__}{args} -> {exc}")
        return []


# Streamlit: web arayüzü (static/), sürüm bilgisi ve iç modüller
datas += safe(collect_data_files, "streamlit")
datas += safe(copy_metadata, "streamlit")
hiddenimports += safe(collect_submodules, "streamlit")

# Grafik ve fiş kütüphanelerinin veri dosyaları (şemalar, fontlar)
for pkg in ("altair", "reportlab", "qrcode"):
    datas += safe(collect_data_files, pkg)

# Ultralytics: model/tracker yaml dosyaları (bytetrack.yaml vb.) ve alt modüller
datas += safe(collect_data_files, "ultralytics")
hiddenimports += safe(collect_submodules, "ultralytics")

# importlib.metadata ile sürümü sorgulanan paketler
for pkg in ("ultralytics", "torch", "torchvision", "altair", "pandas", "numpy", "pillow",
            "opencv-python", "pyarrow", "reportlab", "qrcode", "lap"):
    datas += safe(copy_metadata, pkg)

a = Analysis(
    [str(ROOT / "launcher.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "pytest", "IPython", "jupyter", "notebook"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="SmartCheckout",
    debug=False,
    strip=False,
    upx=False,  # UPX torch/CUDA DLL'lerini bozabilir
    console=True,  # adres ve hata mesajları görünsün; pencereyi kapatmak uygulamayı kapatır
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="SmartCheckout",
)
