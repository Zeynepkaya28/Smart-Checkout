"""Smart Checkout başlatıcısı: Streamlit sunucusunu başlatır ve tarayıcıyı açar.

Hem normal Python ile hem de PyInstaller ile üretilen SmartCheckout.exe içinden çalışır:
    python launcher.py               # geliştirme
    SmartCheckout.exe                   # paketlenmiş sürüm (çift tıklama)
    SmartCheckout.exe --port 8600 --no-browser

Konsol penceresini kapatmak (veya Ctrl+C) uygulamayı durdurur.
"""
from __future__ import annotations

import argparse
import multiprocessing
import os
import socket
import sys
import threading
import time
import webbrowser

import config


def _pyinstaller_imports() -> None:  # pragma: no cover - hiç çağrılmaz
    """Bu fonksiyon ÇAĞRILMAZ.

    Streamlit app.py'yi çalışma anında dosyadan okuduğu için PyInstaller app.py'nin
    içe aktardığı paketleri kendiliğinden göremez. Buradaki importlar, PyInstaller'ın
    bu paketleri .exe'ye dahil etmesi için yazılmıştır.
    """
    import altair  # noqa: F401
    import catalog  # noqa: F401
    import cv2  # noqa: F401
    import detector  # noqa: F401
    import lap  # noqa: F401
    import matplotlib  # noqa: F401
    import numpy  # noqa: F401
    import pandas  # noqa: F401
    import PIL  # noqa: F401
    import qrcode  # noqa: F401
    import receipt  # noqa: F401
    import reportlab  # noqa: F401
    import torch  # noqa: F401
    import ultralytics  # noqa: F401


def find_free_port(start: int = 8501, attempts: int = 30) -> int:
    """start'tan itibaren boş bir TCP portu bulur."""
    for port in range(start, start + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"{start}-{start + attempts - 1} aralığında boş port bulunamadı.")


def open_browser_when_ready(url: str, port: int, timeout: float = 120.0) -> None:
    """Sunucu yanıt vermeye başlayınca varsayılan tarayıcıda uygulamayı açar."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                break
        except OSError:
            time.sleep(0.5)
    else:
        print(f"[UYARI] Sunucu {timeout:.0f} sn içinde açılmadı. Tarayıcıda elle açın: {url}")
        return
    time.sleep(1.0)  # ilk sayfanın hazır olması için kısa bekleme
    webbrowser.open(url)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smart Checkout başlatıcısı")
    parser.add_argument("--port", type=int, default=0, help="Sunucu portu (varsayılan: 8501'den itibaren ilk boş port)")
    parser.add_argument("--no-browser", action="store_true", help="Tarayıcıyı otomatik açma")
    parser.add_argument("--lan", action="store_true",
                        help="Yerel ağdaki diğer cihazlardan erişime izin ver (varsayılan: yalnızca bu bilgisayar)")
    return parser.parse_args()


def main() -> None:
    multiprocessing.freeze_support()  # Windows'ta paketlenmiş torch için gerekli
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    # Model (.pt) indirmeleri, sales.csv ve fisler/ .exe'nin yanında oluşsun
    os.chdir(config.BASE_DIR)
    config.ensure_user_files()

    app_path = config.BUNDLE_DIR / "app.py"
    if not app_path.is_file():
        sys.exit(f"[HATA] app.py bulunamadı: {app_path}")

    port = args.port or find_free_port()
    address = "0.0.0.0" if args.lan else "127.0.0.1"
    url = f"http://localhost:{port}"

    print("=" * 60)
    print(f"  🛒 {config.APP_NAME} – {config.MARKET_NAME}")
    print(f"  Adres   : {url}")
    print(f"  Klasör  : {config.BASE_DIR}")
    print("  Kapatmak için bu pencereyi kapatın veya Ctrl+C'ye basın.")
    print("=" * 60)

    if not args.no_browser:
        threading.Thread(target=open_browser_when_ready, args=(url, port), daemon=True).start()

    from streamlit.web import cli as stcli

    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
        f"--server.port={port}",
        f"--server.address={address}",
        "--server.headless=true",  # tarayıcıyı biz açıyoruz
        "--server.fileWatcherType=none",
        "--global.developmentMode=false",  # .exe içinde zorunlu
        "--browser.gatherUsageStats=false",
        "--client.toolbarMode=minimal",  # "Deploy" düğmesini gizle
        "--theme.base=light",
    ]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
