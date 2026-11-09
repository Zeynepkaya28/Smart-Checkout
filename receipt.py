"""Fiş hesaplama (KDV dahil fiyatlar) ve fişi metin / PDF / PNG olarak üretme.

Para hesapları float hatalarından kaçınmak için Decimal ile yapılır.
Ağır kütüphaneler (reportlab, Pillow, qrcode) yalnızca ilgili fonksiyon
çağrıldığında içe aktarılır; böylece testler sadece standart kütüphaneyle çalışır.
"""
from __future__ import annotations

import csv
import io
import secrets
from dataclasses import dataclass, field
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Callable, Iterable, Literal, Mapping

import config
from catalog import Product

CENT = Decimal("0.01")


# ---------------------------------------------------------------------------
# Para yardımcıları
# ---------------------------------------------------------------------------
def to_decimal(value: Decimal | float | int | str) -> Decimal:
    """Sayıyı Decimal'e çevirir (float için str üzerinden; 0.1 -> 0.1)."""
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def money(value: Decimal | float | int | str) -> Decimal:
    """Kuruşa yuvarlar (0,005 -> 0,01)."""
    return to_decimal(value).quantize(CENT, rounding=ROUND_HALF_UP)


def format_tl(amount: Decimal | float | int, symbol: str = config.CURRENCY_SYMBOL) -> str:
    """Türkçe para formatı: 1234.5 -> '1.234,50 ₺'."""
    value = money(amount)
    sign = "-" if value < 0 else ""
    text = f"{abs(value):,.2f}"  # '1,234.50'
    text = text.replace(",", "X").replace(".", ",").replace("X", ".")  # '1.234,50'
    return f"{sign}{text} {symbol}".rstrip()


# ---------------------------------------------------------------------------
# Veri modelleri
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CartLine:
    """Fişteki tek satır. unit_price KDV dahildir."""

    code: str  # oiv7_name
    name: str  # Türkçe ad
    qty: int
    unit_price: Decimal
    kdv_rate: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "unit_price", money(self.unit_price))
        if int(self.qty) != self.qty or self.qty <= 0:
            raise ValueError(f"'{self.name}' için adet pozitif tam sayı olmalı: {self.qty}")
        if self.unit_price < 0:
            raise ValueError(f"'{self.name}' için fiyat negatif olamaz.")
        if not 0 <= self.kdv_rate <= 100:
            raise ValueError(f"'{self.name}' için KDV oranı 0-100 arasında olmalı.")

    @property
    def total(self) -> Decimal:
        return money(self.unit_price * self.qty)


@dataclass(frozen=True)
class KdvGroup:
    """Bir KDV oranına ait toplamlar."""

    rate: int
    gross: Decimal  # KDV dahil toplam
    net: Decimal  # matrah (KDV hariç)
    kdv: Decimal  # KDV tutarı


@dataclass(frozen=True)
class Receipt:
    """Hesaplanmış fiş."""

    receipt_no: str
    issued_at: datetime
    lines: tuple[CartLine, ...]
    kdv_groups: tuple[KdvGroup, ...]
    total: Decimal
    market_name: str = config.MARKET_NAME

    @property
    def net_total(self) -> Decimal:
        return sum((g.net for g in self.kdv_groups), Decimal("0.00"))

    @property
    def kdv_total(self) -> Decimal:
        return sum((g.kdv for g in self.kdv_groups), Decimal("0.00"))

    @property
    def item_count(self) -> int:
        return sum(line.qty for line in self.lines)


# ---------------------------------------------------------------------------
# Hesaplama
# ---------------------------------------------------------------------------
def kdv_from_gross(gross: Decimal | float | int, rate: int) -> tuple[Decimal, Decimal]:
    """KDV dahil tutardan (matrah, KDV) hesaplar. KDV = brüt - matrah olduğundan toplam her zaman tutar."""
    gross_d = money(gross)
    net = money(gross_d * 100 / (100 + rate))
    return net, gross_d - net


def compute_kdv_groups(lines: Iterable[CartLine]) -> list[KdvGroup]:
    """Satırları KDV oranına göre gruplar; oranlar küçükten büyüğe sıralanır."""
    gross_by_rate: dict[int, Decimal] = {}
    for line in lines:
        gross_by_rate[line.kdv_rate] = gross_by_rate.get(line.kdv_rate, Decimal("0.00")) + line.total
    groups: list[KdvGroup] = []
    for rate in sorted(gross_by_rate):
        gross = money(gross_by_rate[rate])
        net, kdv = kdv_from_gross(gross, rate)
        groups.append(KdvGroup(rate=rate, gross=gross, net=net, kdv=kdv))
    return groups


def generate_receipt_no(now: datetime | None = None) -> str:
    """Benzersiz fiş numarası: YM-20261109-153045-A1B2."""
    now = now or datetime.now()
    return f"YM-{now:%Y%m%d-%H%M%S}-{secrets.token_hex(2).upper()}"


def build_receipt(
    lines: Iterable[CartLine],
    receipt_no: str | None = None,
    issued_at: datetime | None = None,
    market_name: str = config.MARKET_NAME,
) -> Receipt:
    """Satırlardan KDV dökümü ve genel toplamı hesaplanmış fiş oluşturur."""
    issued_at = issued_at or datetime.now()
    line_tuple = tuple(lines)
    groups = tuple(compute_kdv_groups(line_tuple))
    total = money(sum((line.total for line in line_tuple), Decimal("0.00")))
    return Receipt(
        receipt_no=receipt_no or generate_receipt_no(issued_at),
        issued_at=issued_at,
        lines=line_tuple,
        kdv_groups=groups,
        total=total,
        market_name=market_name,
    )


def lines_from_cart(cart: Mapping[str, int], products: Mapping[str, Product]) -> list[CartLine]:
    """Sepet sözlüğünü (oiv7_name -> adet) katalog fiyatlarıyla fiş satırlarına çevirir.

    Adedi 0 olan veya katalogda bulunmayan ürünler atlanır.
    """
    lines: list[CartLine] = []
    for code, qty in cart.items():
        product = products.get(code)
        if product is None or qty <= 0:
            continue
        lines.append(
            CartLine(
                code=code,
                name=product.tr_name,
                qty=int(qty),
                unit_price=to_decimal(product.price),
                kdv_rate=int(product.kdv),
            )
        )
    return lines


# ---------------------------------------------------------------------------
# Fiş düzeni (metin, PDF ve PNG aynı satırları kullanır)
# ---------------------------------------------------------------------------
RowKind = Literal["center", "left", "lr", "sep", "blank"]
RowSize = Literal["small", "normal", "large"]


@dataclass(frozen=True)
class Row:
    kind: RowKind
    left: str = ""
    right: str = ""
    bold: bool = False
    size: RowSize = "normal"


def build_rows(receipt: Receipt, symbol: str = config.CURRENCY_SYMBOL) -> list[Row]:
    """Fişi çizime hazır satırlara böler."""

    def fmt(amount: Decimal) -> str:
        return format_tl(amount, symbol)

    rows: list[Row] = [
        Row("center", receipt.market_name, bold=True, size="large"),
        Row("center", f"{config.APP_NAME} · Kasiyersiz Satış", size="small"),
        Row("blank"),
        Row("lr", "Tarih:", receipt.issued_at.strftime("%d.%m.%Y %H:%M:%S")),
        Row("lr", "Fiş No:", receipt.receipt_no),
        Row("sep"),
        Row("lr", "ÜRÜN", "TUTAR", bold=True),
    ]
    for line in receipt.lines:
        rows.append(Row("left", line.name))
        rows.append(Row("lr", f"  {line.qty} x {fmt(line.unit_price)}", fmt(line.total)))
    if not receipt.lines:
        rows.append(Row("center", "(sepet boş)", size="small"))

    rows += [
        Row("sep"),
        Row("lr", "Toplam ürün adedi", str(receipt.item_count)),
        Row("lr", "Ara toplam (KDV hariç)", fmt(receipt.net_total)),
        Row("lr", "Toplam KDV", fmt(receipt.kdv_total)),
        Row("sep"),
        Row("center", "KDV DÖKÜMÜ", bold=True, size="small"),
    ]
    for g in receipt.kdv_groups:
        rows.append(Row("lr", f"%{g.rate}  Matrah {fmt(g.net)}", f"KDV {fmt(g.kdv)}", size="small"))
    rows += [
        Row("sep"),
        Row("lr", "GENEL TOPLAM", fmt(receipt.total), bold=True, size="large"),
        Row("sep"),
        Row("center", "Fiyatlara KDV dahildir.", size="small"),
        Row("center", "Bizi tercih ettiğiniz için teşekkürler!", size="small"),
        Row("center", "DEMO fiş – mali değeri yoktur.", size="small"),
    ]
    return rows


def render_text(receipt: Receipt, width: int = 42) -> str:
    """Fişi sabit genişlikli düz metne çevirir (önizleme ve konsol için)."""
    out: list[str] = []
    for row in build_rows(receipt):
        if row.kind == "sep":
            out.append("-" * width)
        elif row.kind == "blank":
            out.append("")
        elif row.kind == "center":
            out.append(row.left[:width].center(width).rstrip())
        elif row.kind == "left":
            out.append(row.left[:width])
        else:  # lr
            if len(row.left) + len(row.right) + 1 <= width:
                out.append(row.left + " " * (width - len(row.left) - len(row.right)) + row.right)
            else:
                out.append(row.left[:width])
                out.append(row.right.rjust(width))
    return "\n".join(out)


# Türkçe karakter desteklemeyen yedek font için dönüşüm tablosu
_ASCII_TR = str.maketrans(
    {"ç": "c", "Ç": "C", "ğ": "g", "Ğ": "G", "ı": "i", "İ": "I", "ö": "o", "Ö": "O",
     "ş": "s", "Ş": "S", "ü": "u", "Ü": "U", "·": "-", "–": "-", "₺": "TL"}
)


def _fit(text: str, max_width: float, measure: Callable[[str], float]) -> str:
    """Metni verilen genişliğe sığacak şekilde '…' ile kısaltır."""
    if measure(text) <= max_width:
        return text
    while text and measure(text + "…") > max_width:
        text = text[:-1]
    return text + "…"


# --- PDF -----------------------------------------------------------------------
def _register_pdf_fonts() -> tuple[str, str, bool]:
    """DejaVuSans'ı reportlab'e kaydeder. (normal_ad, kalın_ad, unicode_destekli) döner."""
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    regular_path, bold_path = config.find_font(False), config.find_font(True)
    if regular_path is None:
        return "Helvetica", "Helvetica-Bold", False
    registered = pdfmetrics.getRegisteredFontNames()
    if "KasaFont" not in registered:
        pdfmetrics.registerFont(TTFont("KasaFont", str(regular_path)))
    if "KasaFont-Bold" not in registered:
        pdfmetrics.registerFont(TTFont("KasaFont-Bold", str(bold_path or regular_path)))
    return "KasaFont", "KasaFont-Bold", True


def render_pdf(receipt: Receipt) -> bytes:
    """Fişi 80 mm genişliğinde termal fiş görünümlü PDF'e çevirir (font gömülü)."""
    from reportlab.lib.units import mm
    from reportlab.pdfbase.pdfmetrics import stringWidth
    from reportlab.pdfgen import canvas

    regular, bold, unicode_ok = _register_pdf_fonts()
    symbol = config.CURRENCY_SYMBOL if unicode_ok else "TL"
    rows = build_rows(receipt, symbol)
    if not unicode_ok:
        rows = [Row(r.kind, r.left.translate(_ASCII_TR), r.right.translate(_ASCII_TR), r.bold, r.size) for r in rows]

    sizes = {"small": 6.5, "normal": 8.0, "large": 12.0}
    page_w, margin = 80 * mm, 4 * mm
    usable = page_w - 2 * margin

    def row_height(row: Row) -> float:
        if row.kind == "sep":
            return 7.0
        if row.kind == "blank":
            return 5.0
        return sizes[row.size] * 1.45

    page_h = 2 * margin + sum(row_height(r) for r in rows)
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=(page_w, page_h))
    pdf.setTitle(f"{receipt.market_name} - {receipt.receipt_no}")
    pdf.setAuthor(receipt.market_name)

    y = page_h - margin
    for row in rows:
        h = row_height(row)
        y -= h
        if row.kind == "sep":
            pdf.setDash(2, 2)
            pdf.setLineWidth(0.5)
            pdf.line(margin, y + h / 2, page_w - margin, y + h / 2)
            pdf.setDash()
            continue
        if row.kind == "blank":
            continue
        font, size = (bold if row.bold else regular), sizes[row.size]
        pdf.setFont(font, size)
        baseline = y + h * 0.28

        def measure(t: str, f: str = font, s: float = size) -> float:
            return stringWidth(t, f, s)

        if row.kind == "center":
            pdf.drawCentredString(page_w / 2, baseline, _fit(row.left, usable, measure))
        elif row.kind == "left":
            pdf.drawString(margin, baseline, _fit(row.left, usable, measure))
        else:  # lr
            right_w = measure(row.right)
            pdf.drawRightString(page_w - margin, baseline, row.right)
            pdf.drawString(margin, baseline, _fit(row.left, usable - right_w - 6, measure))

    pdf.showPage()
    pdf.save()
    return buffer.getvalue()


# --- PNG -----------------------------------------------------------------------
def _pil_font(size: int, bold: bool):  # -> ImageFont
    from PIL import ImageFont

    path = config.find_font(bold)
    if path is None:
        try:
            return ImageFont.load_default(size=size)
        except TypeError:
            return ImageFont.load_default()
    return ImageFont.truetype(str(path), size)


def render_png(receipt: Receipt, width: int = 576) -> bytes:
    """Fişi PNG görsele çevirir (576 px ≈ 80 mm termal yazıcı genişliği)."""
    from PIL import Image, ImageDraw

    unicode_ok = config.find_font(False) is not None
    symbol = config.CURRENCY_SYMBOL if unicode_ok else "TL"
    rows = build_rows(receipt, symbol)
    if not unicode_ok:
        rows = [Row(r.kind, r.left.translate(_ASCII_TR), r.right.translate(_ASCII_TR), r.bold, r.size) for r in rows]

    sizes = {"small": 17, "normal": 22, "large": 32}
    margin = 24
    usable = width - 2 * margin

    def row_height(row: Row) -> int:
        if row.kind == "sep":
            return 18
        if row.kind == "blank":
            return 12
        return int(sizes[row.size] * 1.5)

    height = 2 * margin + sum(row_height(r) for r in rows)
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    y = margin
    for row in rows:
        h = row_height(row)
        if row.kind == "sep":
            for x in range(margin, width - margin, 12):
                draw.line([(x, y + h // 2), (min(x + 6, width - margin), y + h // 2)], fill="black", width=2)
        elif row.kind != "blank":
            font = _pil_font(sizes[row.size], row.bold)

            def measure(t: str, f=font) -> float:
                return draw.textlength(t, font=f)

            ty = y + (h - sizes[row.size]) // 2
            if row.kind == "center":
                text = _fit(row.left, usable, measure)
                draw.text(((width - measure(text)) / 2, ty), text, font=font, fill="black")
            elif row.kind == "left":
                draw.text((margin, ty), _fit(row.left, usable, measure), font=font, fill="black")
            else:
                right_w = measure(row.right)
                draw.text((width - margin - right_w, ty), row.right, font=font, fill="black")
                draw.text((margin, ty), _fit(row.left, usable - right_w - 10, measure), font=font, fill="black")
        y += h

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Demo ödeme QR kodu
# ---------------------------------------------------------------------------
def make_demo_qr(receipt: Receipt) -> bytes:
    """Toplam tutarı ve 'DEMO – gerçek ödeme değildir' uyarısını içeren QR görseli (PNG).

    QR içeriği bilgi amaçlıdır; hiçbir ödeme sistemine bağlı DEĞİLDİR.
    """
    import qrcode
    from PIL import Image, ImageDraw

    payload = (
        "DEMO - GERCEK ODEME DEGILDIR\n"
        f"{receipt.market_name}\nFis: {receipt.receipt_no}\nTutar: {format_tl(receipt.total, 'TL')}"
    )
    qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_M, box_size=8, border=2)
    qr.add_data(payload)
    qr.make(fit=True)
    qr_img = qr.make_image(fill_color="black", back_color="white").convert("RGB")

    unicode_ok = config.find_font(False) is not None
    total_text = format_tl(receipt.total, config.CURRENCY_SYMBOL if unicode_ok else "TL")
    warning = "DEMO – gerçek ödeme değildir" if unicode_ok else "DEMO - gercek odeme degildir"
    title_font, warn_font = _pil_font(30, True), _pil_font(20, True)

    pad, top, bottom = 24, 60, 50
    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    ww = measure.textlength(warning, font=warn_font)
    tw = measure.textlength(total_text, font=title_font)
    w = int(max(qr_img.width, ww, tw) + 2 * pad)
    canvas_img = Image.new("RGB", (w, qr_img.height + top + bottom), "white")
    canvas_img.paste(qr_img, ((w - qr_img.width) // 2, top))
    draw = ImageDraw.Draw(canvas_img)
    draw.text(((w - tw) / 2, 14), total_text, font=title_font, fill="black")
    draw.text(((w - ww) / 2, top + qr_img.height + 12), warning, font=warn_font, fill=(200, 0, 0))

    buffer = io.BytesIO()
    canvas_img.save(buffer, format="PNG")
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Kayıt
# ---------------------------------------------------------------------------
SALES_COLUMNS: tuple[str, ...] = (
    "fis_no", "tarih", "oiv7_name", "urun", "adet", "birim_fiyat", "kdv_orani", "tutar",
)


def append_sale_to_csv(receipt: Receipt, path: Path | str = config.SALES_CSV) -> None:
    """Tamamlanan satışı satır satır sales.csv'ye ekler (dosya yoksa başlıkla oluşturur)."""
    path = Path(path)
    is_new = not path.exists() or path.stat().st_size == 0
    # Excel'in Türkçe karakterleri doğru açması için yeni dosyaya BOM yazılır
    with path.open("a", encoding="utf-8-sig" if is_new else "utf-8", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(SALES_COLUMNS)
        for line in receipt.lines:
            writer.writerow([
                receipt.receipt_no,
                receipt.issued_at.isoformat(timespec="seconds"),
                line.code,
                line.name,
                line.qty,
                str(line.unit_price),
                line.kdv_rate,
                str(line.total),
            ])


def save_receipt_files(receipt: Receipt, directory: Path | str = config.RECEIPTS_DIR) -> tuple[Path, Path]:
    """Fişi PDF ve PNG olarak klasöre kaydeder; dosya yollarını döndürür."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    pdf_path = directory / f"{receipt.receipt_no}.pdf"
    png_path = directory / f"{receipt.receipt_no}.png"
    pdf_path.write_bytes(render_pdf(receipt))
    png_path.write_bytes(render_png(receipt))
    return pdf_path, png_path


@dataclass
class CompletedSale:
    """Ödemesi alınmış satışın özeti (ödeme ekranında gösterim için)."""

    receipt: Receipt
    pdf: bytes
    png: bytes
    files: tuple[Path, Path] | None = None
    errors: list[str] = field(default_factory=list)
