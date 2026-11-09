"""Streamlit'siz hızlı demo: yalnızca OpenCV penceresiyle canlı Smart Checkout.

Kısayollar (pencere seçiliyken):
    S  -> Sepeti tara (son 15 karenin sayımı) ve fişi konsola yazdır
    C  -> Sepeti temizle
    Q  -> Çıkış (ESC de çalışır)

Örnek:
    python live_cam.py --camera 0 --model m --conf 0.35
    python live_cam.py --track          # ByteTrack ile benzersiz ID sayımı
"""
from __future__ import annotations

import argparse
import sys
import time

import cv2

import config
from catalog import CatalogError, load_catalog, products_by_name
from detector import CameraError, Detector, ModelLoadError, ScanBuffer, annotate, draw_panel, open_camera
from receipt import build_receipt, format_tl, lines_from_cart, render_text

WINDOW_TITLE = "Smart Checkout - S: tara | C: temizle | Q: cikis"  # Windows pencere başlığında ASCII daha güvenli


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smart Checkout – OpenCV canlı demo")
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX, help="Kamera indeksi (varsayılan: 0)")
    parser.add_argument("--model", default=config.DEFAULT_MODEL_SIZE, choices=list(config.MODEL_SIZES),
                        help="Model boyutu: n/s/m/l/x (varsayılan: m)")
    parser.add_argument("--conf", type=float, default=config.DEFAULT_CONF, help="Güven eşiği")
    parser.add_argument("--iou", type=float, default=config.DEFAULT_IOU, help="IoU eşiği")
    parser.add_argument("--track", action="store_true", help="ByteTrack ile benzersiz ID sayımı kullan")
    return parser.parse_args()


def main() -> int:
    # Windows konsolunda Türkçe karakter ve ₺ için UTF-8 çıktı
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    try:
        products = load_catalog(config.PRODUCTS_JSON)
    except CatalogError as exc:
        print(f"[HATA] Ürün kataloğu okunamadı: {exc}")
        return 1
    by_name = products_by_name(products)

    model_name = config.model_name(args.model)
    print(f"[BİLGİ] Model yükleniyor: {model_name} (ilk çalıştırmada indirilecek)...")
    try:
        detector = Detector(model_name, products)
    except ModelLoadError as exc:
        print(f"[HATA] {exc}")
        return 1
    print(f"[BİLGİ] Cihaz: {'GPU (CUDA)' if detector.device != 'cpu' else 'CPU'}")

    try:
        cap = open_camera(args.camera)
    except CameraError as exc:
        print(f"[HATA] {exc}")
        return 1

    buffer = ScanBuffer(config.SCAN_WINDOW)
    cart: dict[str, int] = {}
    message, message_until = "", 0.0
    fps = 0.0
    print("[BİLGİ] Hazır. S = tara, C = temizle, Q = çıkış")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[HATA] Kameradan görüntü alınamadı.")
                break
            t0 = time.perf_counter()
            detections = detector.detect(frame, args.conf, args.iou, track=args.track)
            buffer.add(detections)
            out = annotate(frame, detections)
            fps = 0.9 * fps + 0.1 * (1.0 / max(time.perf_counter() - t0, 1e-6))

            # Sağ üstte sepet özeti, sol üstte durum
            receipt = build_receipt(lines_from_cart(cart, by_name))
            panel = [f"SEPET  ({receipt.item_count} ürün)"]
            panel += [f"{line.qty} x {line.name}  {format_tl(line.total)}" for line in receipt.lines[:12]]
            if len(receipt.lines) > 12:
                panel.append(f"... +{len(receipt.lines) - 12} satır")
            panel.append(f"TOPLAM: {format_tl(receipt.total)}")
            out = draw_panel(out, [f"{fps:.1f} FPS · {'ByteTrack' if args.track else 'Medyan'}",
                                   "S: tara  C: temizle  Q: çıkış"])
            out = draw_panel(out, panel, origin=(max(10, out.shape[1] - 430), 10))
            if message and time.time() < message_until:
                out = draw_panel(out, [message], origin=(10, out.shape[0] - 60))

            cv2.imshow(WINDOW_TITLE, out)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("s"), ord("S")):
                cart = buffer.counts(tracked=args.track)
                receipt = build_receipt(lines_from_cart(cart, by_name))
                print("\n" + render_text(receipt) + "\n")
                message = f"Tarandı: {receipt.item_count} ürün, {format_tl(receipt.total)}"
                message_until = time.time() + 3
            elif key in (ord("c"), ord("C")):
                cart.clear()
                buffer.clear()
                detector.reset_tracker()
                print("[BİLGİ] Sepet temizlendi.")
                message, message_until = "Sepet temizlendi", time.time() + 2
            # Pencere X ile kapatıldıysa çık
            if cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
