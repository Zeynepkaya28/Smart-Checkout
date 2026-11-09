"""Marka tanıma için kendi fotoğraflarınızla YOLO fine-tune iskeleti.

Hazır OIV7 modeli ürünleri TÜRÜNE göre tanır (süt, konserve, şişe) ama markayı
ayırt etmez. Markaları (ör. "A marka süt 1 L" ile "B marka süt 1 L") ayırmak
için kendi etiketli fotoğraflarınızla modeli yeniden eğitebilirsiniz.

Adımlar:
  1) Klasör iskeletini oluştur:
       python train_custom.py --init --classes marka_a_sut_1l marka_b_sut_1l marka_c_kola_330ml
  2) Fotoğrafları images/train ve images/val klasörlerine koyun
     (sınıf başına en az 100-200 fotoğraf, farklı açı/ışık/arka plan).
  3) Etiketleyin (YOLO formatı). Ücretsiz araçlar: Label Studio, CVAT, Roboflow.
     Her görüntü için aynı adlı .txt dosyası labels/... altında olmalı:
       <sınıf_id> <x_merkez> <y_merkez> <genişlik> <yükseklik>   (0-1 arası normalize)
  4) Eğitin:
       python train_custom.py --data datasets/marka_urunler/data.yaml --epochs 100
  5) Uygulamada kullanın:
       set SMART_CHECKOUT_MODEL=runs\\marka\\marka_v1\\weights\\best.pt
       streamlit run app.py
     ve products.json içindeki "oiv7_name" alanlarına yeni sınıf adlarını yazın.

Veri seti klasör yapısı:
  datasets/marka_urunler/
  ├─ data.yaml
  ├─ images/
  │  ├─ train/   foto_0001.jpg, foto_0002.jpg, ...
  │  ├─ val/     foto_0101.jpg, ...
  │  └─ test/    (isteğe bağlı)
  └─ labels/
     ├─ train/   foto_0001.txt, foto_0002.txt, ...
     ├─ val/     foto_0101.txt, ...
     └─ test/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import config

IMAGE_EXTS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
DEFAULT_ROOT = config.BASE_DIR / "datasets" / "marka_urunler"
EXAMPLE_CLASSES = ["marka_a_sut_1l", "marka_b_sut_1l", "marka_a_konserve_bezelye", "marka_c_su_05l"]


def init_dataset(root: Path, classes: list[str]) -> Path:
    """Boş veri seti klasörlerini ve data.yaml şablonunu oluşturur."""
    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)

    data_yaml = root / "data.yaml"
    if data_yaml.exists():
        print(f"[BİLGİ] {data_yaml} zaten var, üzerine yazılmadı.")
        return data_yaml

    names_block = "\n".join(f"  {i}: {name}" for i, name in enumerate(classes))
    data_yaml.write_text(
        "# Smart Checkout – marka tanıma veri seti\n"
        f"path: {root.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test  # isteğe bağlı\n\n"
        "names:\n"
        f"{names_block}\n",
        encoding="utf-8",
    )
    (root / "OKUBENI.txt").write_text(
        "Her görüntü için labels/<split>/ altında aynı adlı bir .txt dosyası olmalı.\n"
        "Her satır bir nesne: <sınıf_id> <x_merkez> <y_merkez> <genişlik> <yükseklik>\n"
        "Koordinatlar görüntü boyutuna bölünmüş (0-1 arası) değerlerdir.\n"
        "Örnek: 0 0.512 0.430 0.210 0.380\n",
        encoding="utf-8",
    )
    print(f"[OK] Veri seti iskeleti oluşturuldu: {root}")
    print(f"     Sınıflar: {', '.join(classes)}")
    return data_yaml


def check_dataset(data_yaml: Path) -> None:
    """Eğitime başlamadan önce temel kontroller yapar."""
    if not data_yaml.is_file():
        sys.exit(f"[HATA] {data_yaml} bulunamadı. Önce: python train_custom.py --init")
    root = data_yaml.parent
    for split in ("train", "val"):
        images = [p for p in (root / "images" / split).glob("*") if p.suffix.lower() in IMAGE_EXTS]
        labels = list((root / "labels" / split).glob("*.txt"))
        print(f"[BİLGİ] {split}: {len(images)} görüntü, {len(labels)} etiket dosyası")
        if not images:
            sys.exit(f"[HATA] images/{split} klasörü boş. Fotoğrafları ekleyip tekrar deneyin.")
        missing = [p.name for p in images if not (root / "labels" / split / f"{p.stem}.txt").exists()]
        if missing:
            print(f"[UYARI] {len(missing)} görüntünün etiketi yok (ör. {missing[0]}); bunlar arka plan sayılır.")


def train(args: argparse.Namespace) -> None:
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("[HATA] ultralytics kurulu değil: pip install -r requirements.txt")

    from detector import select_device

    data_yaml = Path(args.data)
    check_dataset(data_yaml)
    device = select_device()
    print(f"[BİLGİ] Cihaz: {'GPU' if device != 'cpu' else 'CPU (yavaş olacak)'} · Başlangıç modeli: {args.model}")

    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=str(config.BASE_DIR / "runs" / "marka"),
        name=args.name,
        patience=args.patience,
        workers=args.workers,  # Windows'ta yüksek değer sorun çıkarabilir
        plots=True,
    )
    best = Path(model.trainer.best)
    print(f"\n[OK] Eğitim bitti. En iyi ağırlıklar: {best}")

    metrics = YOLO(str(best)).val(data=str(data_yaml), device=device, verbose=False)
    print(f"[BİLGİ] Doğrulama mAP50: {metrics.box.map50:.3f} · mAP50-95: {metrics.box.map:.3f}")
    print("\nUygulamada kullanmak için (PowerShell):")
    print(f'  $env:SMART_CHECKOUT_MODEL = "{best}"')
    print("  streamlit run app.py")
    print("ve products.json içindeki 'oiv7_name' alanlarını data.yaml'daki sınıf adlarıyla eşleştirin.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Marka tanıma için YOLO fine-tune")
    p.add_argument("--init", action="store_true", help="Veri seti klasör iskeletini ve data.yaml'ı oluştur")
    p.add_argument("--root", default=str(DEFAULT_ROOT), help="Veri seti kök klasörü (--init için)")
    p.add_argument("--classes", nargs="+", default=EXAMPLE_CLASSES, help="Sınıf adları (--init için)")
    p.add_argument("--data", default=str(DEFAULT_ROOT / "data.yaml"), help="data.yaml yolu")
    p.add_argument("--model", default="yolov8m.pt",
                   help="Başlangıç ağırlıkları (yolov8m.pt = COCO; yolov8m-oiv7.pt da kullanılabilir)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=-1, help="-1 = GPU belleğine göre otomatik")
    p.add_argument("--patience", type=int, default=20, help="Gelişme olmazsa erken durdurma (epoch)")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--name", default="marka_v1", help="Çıktı klasörü adı (runs/marka/<ad>)")
    return p.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    if args.init:
        init_dataset(Path(args.root), args.classes)
        return
    train(args)


if __name__ == "__main__":
    main()
