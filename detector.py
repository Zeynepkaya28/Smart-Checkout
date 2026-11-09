"""YOLO modelini yükleme, katalog sınıflarına göre filtreleme, çizim ve sayım."""
from __future__ import annotations

import math
import statistics
import sys
from collections import Counter, deque
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import config
from catalog import Product, match_catalog


class ModelLoadError(Exception):
    """Model indirilemediğinde / yüklenemediğinde fırlatılır (mesaj Türkçe)."""


class CameraError(Exception):
    """Kamera açılamadığında fırlatılır (mesaj Türkçe)."""


@dataclass(frozen=True)
class Detection:
    """Tek bir tespit edilen ürün."""

    cls_id: int
    oiv7_name: str
    tr_name: str
    conf: float
    xyxy: tuple[float, float, float, float]
    track_id: int | None = None


# ---------------------------------------------------------------------------
# Cihaz seçimi
# ---------------------------------------------------------------------------
def select_device() -> int | str:
    """CUDA varsa ilk GPU'yu (0), yoksa 'cpu' döndürür."""
    try:
        import torch

        if torch.cuda.is_available():
            return 0
    except Exception:
        pass
    return "cpu"


def device_description() -> str:
    """Kenar çubuğunda gösterilecek okunur cihaz açıklaması."""
    try:
        import torch

        if torch.cuda.is_available():
            return f"GPU (CUDA) – {torch.cuda.get_device_name(0)}"
        return "CPU (CUDA bulunamadı – PyTorch'un CUDA'lı sürümü kurulu mu?)"
    except Exception:
        return "CPU (PyTorch yüklenemedi)"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def _load_yolo(model_name: str):  # -> ultralytics.YOLO
    """YOLO modelini yükler; ilk çalıştırmada Ultralytics otomatik indirir."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ModelLoadError(
            "'ultralytics' paketi kurulu değil. Sanal ortamı etkinleştirip "
            "'pip install -r requirements.txt' komutunu çalıştırın."
        ) from exc
    try:
        return YOLO(model_name)
    except Exception as exc:  # ağ hatası, bozuk dosya, yanlış ad...
        raise ModelLoadError(
            f"'{model_name}' modeli indirilemedi veya yüklenemedi.\n"
            "• İnternet bağlantınızı kontrol edin (ilk çalıştırmada model GitHub'dan indirilir).\n"
            "• Alternatif: dosyayı https://github.com/ultralytics/assets/releases adresinden elle "
            "indirip proje klasörüne koyun.\n"
            "• Yarım inmiş bir .pt dosyası varsa silip tekrar deneyin.\n"
            f"Ayrıntı: {exc}"
        ) from exc


class Detector:
    """Katalogdaki ürünlerle sınırlandırılmış YOLO dedektörü."""

    def __init__(self, model_name: str, products: Iterable[Product], device: int | str | None = None) -> None:
        self.model_name = model_name
        self.device: int | str = select_device() if device is None else device
        self.half: bool = self.device != "cpu"  # GPU'da FP16 ile daha hızlı
        self.model = _load_yolo(model_name)

        match = match_catalog(self.model.names, products, warn=True)
        self.class_map: dict[int, Product] = match.class_map
        self.class_ids: list[int] = sorted(match.class_map)
        self.missing: list[tuple[str, list[str]]] = match.missing
        if not self.class_ids:
            raise ModelLoadError(
                "Katalogdaki ürünlerin hiçbiri modelin sınıflarıyla eşleşmedi. "
                "products.json içindeki 'oiv7_name' değerlerini kontrol edin."
            )

    def detect(self, frame: np.ndarray, conf: float, iou: float, track: bool = False) -> list[Detection]:
        """Bir BGR karede yalnızca katalog sınıflarını tespit eder.

        track=True ise ByteTrack ile kareler arası takip yapılır ve her
        tespite benzersiz bir track_id atanır.
        """
        kwargs = dict(
            conf=conf,
            iou=iou,
            classes=self.class_ids,  # diğer ~530 sınıf yok sayılır
            device=self.device,
            half=self.half,
            verbose=False,
        )
        if track:
            results = self.model.track(frame, persist=True, tracker="bytetrack.yaml", **kwargs)
        else:
            results = self.model.predict(frame, **kwargs)
        if not results:
            return []

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []
        cls_arr = boxes.cls.int().cpu().tolist()
        conf_arr = boxes.conf.cpu().tolist()
        xyxy_arr = boxes.xyxy.cpu().tolist()
        ids_arr: list[int | None] = (
            boxes.id.int().cpu().tolist() if track and boxes.id is not None else [None] * len(cls_arr)
        )

        detections: list[Detection] = []
        for cls_id, score, xyxy, tid in zip(cls_arr, conf_arr, xyxy_arr, ids_arr):
            product = self.class_map.get(cls_id)
            if product is None:  # güvenlik: filtre dışı sınıf gelirse at
                continue
            detections.append(
                Detection(
                    cls_id=cls_id,
                    oiv7_name=product.oiv7_name,
                    tr_name=product.tr_name,
                    conf=float(score),
                    xyxy=(xyxy[0], xyxy[1], xyxy[2], xyxy[3]),
                    track_id=tid,
                )
            )
        return detections

    def reset_tracker(self) -> None:
        """ByteTrack durumunu sıfırlar (mod değişince eski ID'ler karışmasın)."""
        predictor = getattr(self.model, "predictor", None)
        for tracker in getattr(predictor, "trackers", None) or []:
            try:
                tracker.reset()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Kamera
# ---------------------------------------------------------------------------
def open_camera(index: int, width: int = 1280, height: int = 720) -> cv2.VideoCapture:
    """Kamerayı açar. Windows'ta DirectShow daha hızlı açıldığı için önce onu dener."""
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY] if sys.platform.startswith("win") else [cv2.CAP_ANY]
    for backend in backends:
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        ok, _ = cap.read()
        if ok:
            return cap
        cap.release()
    raise CameraError(
        f"Kamera açılamadı (indeks {index}).\n"
        "• Kamera başka bir uygulama (Zoom, Teams, tarayıcı) tarafından kullanılıyor olabilir.\n"
        "• Kenar çubuğundan farklı bir kamera indeksi (0, 1, 2...) deneyin.\n"
        "• Windows Ayarlar > Gizlilik > Kamera bölümünde masaüstü uygulamalarına izin verin."
    )


# ---------------------------------------------------------------------------
# Sayım
# ---------------------------------------------------------------------------
class ScanBuffer:
    """Son N karenin tespitlerini tutar; aynı ürünün iki kez sayılmasını önler.

    İki yöntem vardır:
      • median: her sınıf için karelerdeki adetlerin medyanı (titreşimlere dayanıklı)
      • tracked: ByteTrack ID'leri ile benzersiz nesne sayısı
    """

    def __init__(self, window: int = config.SCAN_WINDOW) -> None:
        self.window = window
        self._frames: deque[list[Detection]] = deque(maxlen=window)

    def __len__(self) -> int:
        return len(self._frames)

    def add(self, detections: list[Detection]) -> None:
        self._frames.append(list(detections))

    def clear(self) -> None:
        self._frames.clear()

    def _per_frame_counts(self) -> dict[str, list[int]]:
        names = {d.oiv7_name for frame in self._frames for d in frame}
        frame_counters = [Counter(d.oiv7_name for d in frame) for frame in self._frames]
        return {name: [c.get(name, 0) for c in frame_counters] for name in names}

    def counts_median(self) -> dict[str, int]:
        """Her sınıf için kare başına adetlerin medyanı (0.5 yukarı yuvarlanır)."""
        result: dict[str, int] = {}
        for name, per_frame in self._per_frame_counts().items():
            value = int(math.floor(statistics.median(per_frame) + 0.5))
            if value > 0:
                result[name] = value
        return result

    def counts_tracked(self) -> dict[str, int]:
        """Her sınıf için pencere içindeki benzersiz takip ID'si sayısı.

        Bir sınıf için hiç ID yoksa (takip henüz oturmadıysa) medyana düşer.
        """
        ids: dict[str, set[int]] = {}
        for frame in self._frames:
            for d in frame:
                if d.track_id is not None:
                    ids.setdefault(d.oiv7_name, set()).add(d.track_id)
        median = self.counts_median()
        result = {name: len(s) for name, s in ids.items() if s}
        for name, value in median.items():
            result.setdefault(name, value)
        return result

    def counts(self, tracked: bool = False) -> dict[str, int]:
        return self.counts_tracked() if tracked else self.counts_median()


def count_detections(detections: Iterable[Detection]) -> dict[str, int]:
    """Tek bir görüntüdeki tespitleri sınıf başına sayar (fotoğraf modu)."""
    return dict(Counter(d.oiv7_name for d in detections))


# ---------------------------------------------------------------------------
# Çizim (Türkçe karakterler için PIL + TTF font)
# ---------------------------------------------------------------------------
_PALETTE_BGR: list[tuple[int, int, int]] = [
    (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255), (49, 210, 207),
    (10, 249, 72), (23, 204, 146), (134, 219, 61), (52, 147, 26), (187, 212, 0),
    (168, 153, 44), (255, 194, 0), (147, 69, 52), (255, 115, 100), (236, 24, 0),
    (255, 56, 132), (133, 0, 82), (255, 56, 203), (200, 149, 255), (199, 55, 255),
]


def _color(cls_id: int) -> tuple[int, int, int]:
    return _PALETTE_BGR[cls_id % len(_PALETTE_BGR)]


@lru_cache(maxsize=16)
def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = config.find_font(bold)
    if path is None:
        try:
            return ImageFont.load_default(size=size)
        except TypeError:  # eski Pillow
            return ImageFont.load_default()
    return ImageFont.truetype(str(path), size)


def annotate(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Tespit kutularını ve Türkçe etiketleri BGR kareye çizer (yeni kopya döner)."""
    out = frame.copy()
    if not detections:
        return out
    thickness = max(2, out.shape[0] // 360)
    for d in detections:
        x1, y1, x2, y2 = (int(v) for v in d.xyxy)
        cv2.rectangle(out, (x1, y1), (x2, y2), _color(d.cls_id), thickness)

    # OpenCV putText Türkçe karakter (ş, ğ, ı, İ) çizemez; etiketleri PIL ile çiz.
    img = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    font = _font(max(14, out.shape[0] // 30), bold=True)
    for d in detections:
        label = f"{d.tr_name} {d.conf:.0%}"
        if d.track_id is not None:
            label += f" #{d.track_id}"
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        tw, th = right - left, bottom - top
        x1, y1 = int(d.xyxy[0]), int(d.xyxy[1])
        ty = y1 - th - 8 if y1 - th - 8 >= 0 else y1
        b, g, r = _color(d.cls_id)
        draw.rectangle([x1, ty, x1 + tw + 8, ty + th + 8], fill=(r, g, b))
        draw.text((x1 + 4, ty + 4 - top), label, font=font, fill=(255, 255, 255))
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def draw_panel(frame: np.ndarray, lines: list[str], origin: tuple[int, int] = (10, 10)) -> np.ndarray:
    """Karenin köşesine yarı saydam bilgi paneli çizer (live_cam.py için)."""
    if not lines:
        return frame
    font = _font(max(14, frame.shape[0] // 40))
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    line_h = int(font.size * 1.35) if hasattr(font, "size") else 18
    width = max(int(draw.textlength(t, font=font)) for t in lines) + 20
    x, y = origin
    draw.rectangle([x, y, x + width, y + line_h * len(lines) + 12], fill=(0, 0, 0, 160))
    for i, text in enumerate(lines):
        draw.text((x + 10, y + 6 + i * line_h), text, font=font, fill=(255, 255, 255, 255))
    merged = Image.alpha_composite(img, overlay).convert("RGB")
    return cv2.cvtColor(np.asarray(merged), cv2.COLOR_RGB2BGR)
