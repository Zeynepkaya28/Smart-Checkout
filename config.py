"""Uygulama genel ayarları ve dosya yolları.

Ayarların bir kısmı ortam değişkenleriyle değiştirilebilir:
    SMART_CHECKOUT_MODEL_SIZE  -> n / s / m / l / x (varsayılan: m)
    SMART_CHECKOUT_MODEL       -> Tam model yolu (ör. kendi eğittiğiniz best.pt). Verilirse boyut seçimini ezer.
    SMART_CHECKOUT_CAMERA      -> Varsayılan kamera indeksi (varsayılan: 0)
"""
from __future__ import annotations

import os
import shutil
import sys
from functools import lru_cache
from pathlib import Path

# --- Dosya yolları -----------------------------------------------------------
# .exe (PyInstaller) olarak çalışırken iki farklı klasör vardır:
#   BUNDLE_DIR -> paketin içindeki salt-okunur dosyalar (app.py, varsayılan products.json, fontlar)
#   BASE_DIR   -> .exe'nin yanındaki yazılabilir klasör (sales.csv, fisler/, düzenlenen fiyatlar, model)
# Normal "python ..." çalıştırmada ikisi de proje klasörüdür.
FROZEN: bool = bool(getattr(sys, "frozen", False))
BUNDLE_DIR: Path = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
BASE_DIR: Path = Path(sys.executable).resolve().parent if FROZEN else Path(__file__).resolve().parent
PRODUCTS_JSON: Path = BASE_DIR / "products.json"
SALES_CSV: Path = BASE_DIR / "sales.csv"
RECEIPTS_DIR: Path = BASE_DIR / "fisler"
FONTS_DIR: Path = BASE_DIR / "fonts"


def ensure_user_files() -> None:
    """.exe ilk kez çalıştığında varsayılan products.json'ı .exe'nin yanına kopyalar.

    Böylece fiyat düzenlemeleri paketin içine değil, kullanıcının klasörüne yazılır.
    """
    bundled = BUNDLE_DIR / "products.json"
    if not PRODUCTS_JSON.exists() and bundled.is_file() and bundled != PRODUCTS_JSON:
        shutil.copyfile(bundled, PRODUCTS_JSON)


if FROZEN:
    ensure_user_files()

# --- Market bilgisi ----------------------------------------------------------
APP_NAME: str = "Smart Checkout"  # uygulamanın adı
MARKET_NAME: str = "YOLO Market"  # fişte görünen mağaza adı
CURRENCY_SYMBOL: str = "₺"

# --- Model ayarları ----------------------------------------------------------
# Ultralytics'in Open Images V7 (601 sınıf) üzerinde eğitilmiş hazır modelleri.
MODEL_TEMPLATE: str = "yolov8{size}-oiv7.pt"
MODEL_SIZES: dict[str, str] = {
    "n": "Nano – en hızlı",
    "s": "Small – hızlı",
    "m": "Medium – önerilen",
    "l": "Large – daha doğru",
    "x": "XLarge – en doğru, en yavaş",
}
DEFAULT_MODEL_SIZE: str = os.getenv("SMART_CHECKOUT_MODEL_SIZE", "m").lower()
if DEFAULT_MODEL_SIZE not in MODEL_SIZES:
    DEFAULT_MODEL_SIZE = "m"

# Kendi eğittiğiniz modeli kullanmak için (train_custom.py çıktısı).
CUSTOM_MODEL: str | None = os.getenv("SMART_CHECKOUT_MODEL") or None

# --- Tespit ve sayım ---------------------------------------------------------
DEFAULT_CONF: float = 0.35
DEFAULT_IOU: float = 0.50
SCAN_WINDOW: int = 15  # "Sepeti Tara" için kullanılan son kare sayısı
CAMERA_INDEX: int = int(os.getenv("SMART_CHECKOUT_CAMERA", "0"))
KDV_RATES: tuple[int, ...] = (0, 1, 10, 20)


def model_name(size: str) -> str:
    """Boyut harfinden (n/s/m/l/x) model dosya adını üretir.

    SMART_CHECKOUT_MODEL ortam değişkeni tanımlıysa her zaman onu döndürür.
    """
    if CUSTOM_MODEL:
        return CUSTOM_MODEL
    size = size.lower()
    if size not in MODEL_SIZES:
        raise ValueError(f"Geçersiz model boyutu: {size!r}. Seçenekler: {', '.join(MODEL_SIZES)}")
    return MODEL_TEMPLATE.format(size=size)


@lru_cache(maxsize=4)
def find_font(bold: bool = False) -> Path | None:
    """Türkçe karakterleri ve ₺ simgesini destekleyen bir TTF font bulur.

    Arama sırası:
      1. Proje içindeki fonts/ klasörü (DejaVuSans.ttf)
      2. matplotlib ile birlikte gelen DejaVuSans (ultralytics zaten matplotlib kurar)
      3. Sistem fontları (DejaVu, sonra Windows'ta Arial)
    Hiçbiri yoksa None döner.
    """
    dejavu = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    candidates: list[Path] = [FONTS_DIR / dejavu, BUNDLE_DIR / "fonts" / dejavu]

    try:
        import matplotlib  # yalnızca veri yolunu almak için; pyplot yüklenmez

        candidates.append(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / dejavu)
    except Exception:  # matplotlib yoksa sorun değil
        pass

    if sys.platform.startswith("win"):
        win_fonts = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
        candidates += [win_fonts / dejavu, win_fonts / ("arialbd.ttf" if bold else "arial.ttf")]
    else:
        candidates += [
            Path("/usr/share/fonts/truetype/dejavu") / dejavu,
            Path("/usr/share/fonts/TTF") / dejavu,
            Path("/Library/Fonts/Arial Unicode.ttf"),
        ]

    for path in candidates:
        if path.is_file():
            return path
    return None
