"""Smart Checkout – Streamlit arayüzü.

Çalıştırma:  streamlit run app.py
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Callable

import altair as alt
import cv2
import numpy as np
import pandas as pd
import streamlit as st

import config
from catalog import CatalogError, Product, load_catalog, products_by_name, save_catalog
from detector import (
    CameraError,
    Detection,
    Detector,
    ModelLoadError,
    ScanBuffer,
    annotate,
    count_detections,
    device_description,
    open_camera,
)
from receipt import (
    CompletedSale,
    Receipt,
    append_sale_to_csv,
    build_receipt,
    format_tl,
    generate_receipt_no,
    lines_from_cart,
    make_demo_qr,
    render_pdf,
    render_png,
    render_text,
    save_receipt_files,
)

st.set_page_config(page_title=config.APP_NAME, page_icon="🛒", layout="wide")

PAGE_CHECKOUT = "🛒 Kasa"
PAGE_PAYMENT = "💳 Ödeme"
PAGE_PRICES = "🏷️ Fiyat Düzenleme"
PAGE_HISTORY = "📈 Satış Geçmişi"
PAGES = [PAGE_CHECKOUT, PAGE_PAYMENT, PAGE_PRICES, PAGE_HISTORY]

MODE_LIVE = "Canlı kamera"
MODE_SNAPSHOT = "Fotoğraf çek"
MODE_UPLOAD = "Dosya yükle"


@dataclass(frozen=True)
class Settings:
    """Kenar çubuğundan gelen ayarlar."""

    conf: float
    iou: float
    model_name: str
    camera_index: int
    use_tracking: bool


# ---------------------------------------------------------------------------
# Durum (session_state)
# ---------------------------------------------------------------------------
def init_state() -> None:
    ss = st.session_state
    ss.setdefault("cart", {})  # oiv7_name -> adet (ekleme sırası korunur)
    ss.setdefault("scan_result", {})  # son taramanın sonucu
    ss.setdefault("scan_buffer", ScanBuffer(config.SCAN_WINDOW))
    ss.setdefault("receipt_no", generate_receipt_no())
    ss.setdefault("page", PAGE_CHECKOUT)
    ss.setdefault("last_sale", None)  # CompletedSale | None
    ss.setdefault("payment_done", False)
    ss.setdefault("flash", None)  # bir sonraki çalıştırmada gösterilecek mesaj


# --- Sepet işlemleri (buton callback'leri) ----------------------------------
def add_to_cart(counts: dict[str, int]) -> None:
    cart: dict[str, int] = st.session_state.cart
    for name, qty in counts.items():
        if qty > 0:
            cart[name] = cart.get(name, 0) + int(qty)
    st.session_state.payment_done = False


def add_scan_to_cart() -> None:
    add_to_cart(st.session_state.scan_result)
    st.session_state.scan_result = {}
    st.session_state.flash = ("success", "Tarama sonucu sepete eklendi.")


def replace_cart_with_scan() -> None:
    st.session_state.cart = {k: v for k, v in st.session_state.scan_result.items() if v > 0}
    st.session_state.scan_result = {}
    st.session_state.payment_done = False
    st.session_state.flash = ("success", "Sepet tarama sonucuyla değiştirildi.")


def discard_scan() -> None:
    st.session_state.scan_result = {}


def change_qty(name: str, delta: int) -> None:
    cart: dict[str, int] = st.session_state.cart
    new_qty = cart.get(name, 0) + delta
    if new_qty <= 0:
        cart.pop(name, None)
    else:
        cart[name] = new_qty


def remove_line(name: str) -> None:
    st.session_state.cart.pop(name, None)


def clear_cart() -> None:
    st.session_state.cart = {}
    st.session_state.scan_result = {}
    st.session_state.scan_buffer.clear()
    st.session_state.receipt_no = generate_receipt_no()


def go_to(page: str) -> None:
    st.session_state.page = page


def manual_add() -> None:
    name = st.session_state.get("manual_product")
    qty = int(st.session_state.get("manual_qty", 1))
    if name:
        add_to_cart({name: qty})


def new_customer() -> None:
    st.session_state.payment_done = False
    st.session_state.last_sale = None
    st.session_state.page = PAGE_CHECKOUT


# ---------------------------------------------------------------------------
# Önbellekli kaynaklar
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Model yükleniyor… (ilk çalıştırmada internetten indirilir)")
def get_detector(model_name: str) -> Detector:
    """Modeli uygulama ömrü boyunca bir kez yükler."""
    return Detector(model_name, load_catalog(config.PRODUCTS_JSON))


@st.cache_data(max_entries=16, show_spinner="Fotoğraf analiz ediliyor…")
def detect_image(
    _detector: Detector, model_name: str, image_bytes: bytes, conf: float, iou: float
) -> tuple[np.ndarray, dict[str, int]]:
    """Fotoğraftaki ürünleri bulur; (işaretli RGB görüntü, sayımlar) döner.

    model_name önbellek anahtarına girsin diye parametredir (_detector hash'lenmez).
    """
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Görüntü okunamadı. JPG veya PNG bir dosya deneyin.")
    detections = _detector.detect(img, conf, iou, track=False)
    rgb = cv2.cvtColor(annotate(img, detections), cv2.COLOR_BGR2RGB)
    return rgb, count_detections(detections)


def acquire_camera(index: int) -> cv2.VideoCapture:
    """Kamerayı oturum boyunca açık tutar; indeks değişirse yeniden açar."""
    ss = st.session_state
    cap: cv2.VideoCapture | None = ss.get("_cap")
    if cap is not None and ss.get("_cap_index") == index and cap.isOpened():
        return cap
    release_camera()
    cap = open_camera(index)  # CameraError fırlatabilir
    ss["_cap"], ss["_cap_index"] = cap, index
    return cap


def release_camera() -> None:
    cap: cv2.VideoCapture | None = st.session_state.pop("_cap", None)
    st.session_state.pop("_cap_index", None)
    if cap is not None:
        cap.release()


def get_products() -> list[Product]:
    try:
        return load_catalog(config.PRODUCTS_JSON)
    except CatalogError as exc:
        st.error(f"Ürün kataloğu okunamadı: {exc}")
        st.stop()
        raise  # st.stop() sonrası çalışmaz; tip denetleyici için


def current_receipt(by_name: dict[str, Product]) -> Receipt:
    return build_receipt(lines_from_cart(st.session_state.cart, by_name), receipt_no=st.session_state.receipt_no)


# ---------------------------------------------------------------------------
# Kenar çubuğu
# ---------------------------------------------------------------------------
def sidebar() -> Settings:
    with st.sidebar:
        st.title(f"🛒 {config.APP_NAME}")
        st.caption(f"{config.MARKET_NAME} – kasiyersiz kasa demosu")
        st.radio("Sayfa", PAGES, key="page")
        st.divider()

        st.subheader("⚙️ Tespit ayarları")
        conf = st.slider("Güven eşiği", 0.05, 0.95, config.DEFAULT_CONF, 0.05,
                         help="Bu değerin altındaki tespitler yok sayılır.")
        iou = st.slider("IoU eşiği", 0.10, 0.90, config.DEFAULT_IOU, 0.05,
                        help="Örtüşen kutuların birleştirilme eşiği (NMS).")
        sizes = list(config.MODEL_SIZES)
        if config.CUSTOM_MODEL:
            st.info(f"Özel model kullanılıyor: `{config.CUSTOM_MODEL}`")
            size = config.DEFAULT_MODEL_SIZE
        else:
            size = st.selectbox(
                "Model boyutu",
                sizes,
                index=sizes.index(config.DEFAULT_MODEL_SIZE),
                format_func=lambda s: f"{s.upper()} – {config.MODEL_SIZES[s]}",
            )
        camera_index = int(st.number_input("Kamera indeksi", 0, 10, config.CAMERA_INDEX, 1))
        method = st.radio(
            "Sayım yöntemi",
            ["Medyan (son 15 kare)", "ByteTrack (benzersiz ID)"],
            help="Medyan: son karelerde görülen adetlerin medyanı. "
                 "ByteTrack: her nesneye takip ID'si verip benzersiz ID sayar.",
        )
        st.divider()
        st.caption(f"Cihaz: {device_description()}")
        st.caption(f"Model: `{config.model_name(size)}`")

    return Settings(
        conf=float(conf),
        iou=float(iou),
        model_name=config.model_name(size),
        camera_index=camera_index,
        use_tracking=method.startswith("ByteTrack"),
    )


# ---------------------------------------------------------------------------
# Kasa sayfası
# ---------------------------------------------------------------------------
def page_checkout(settings: Settings, by_name: dict[str, Product]) -> Callable[[], None] | None:
    """Kasa ekranı. Canlı kamera açıksa, en sonda çalıştırılacak döngüyü döndürür."""
    try:
        detector = get_detector(settings.model_name)
    except ModelLoadError as exc:
        st.error(f"Model yüklenemedi.\n\n{exc}")
        return None

    if detector.missing:
        with st.expander(f"⚠️ {len(detector.missing)} katalog ürünü modelde bulunamadı"):
            for name, suggestions in detector.missing:
                st.write(f"• **{name}** → öneri: {', '.join(suggestions) or '—'}")

    # Sayım yöntemi değiştiyse eski kareleri ve takip ID'lerini temizle
    if st.session_state.get("_last_tracking") != settings.use_tracking:
        st.session_state.scan_buffer.clear()
        detector.reset_tracker()
        st.session_state["_last_tracking"] = settings.use_tracking

    live_job: Callable[[], None] | None = None
    left, right = st.columns([3, 2], gap="large")

    with left:
        st.subheader("📷 Görüntü")
        mode = st.radio("Giriş modu", [MODE_LIVE, MODE_SNAPSHOT, MODE_UPLOAD], horizontal=True, key="input_mode")
        if mode == MODE_LIVE:
            live_job = live_camera_controls(detector, settings)
        else:
            release_camera()
            photo_input(detector, settings, mode)
        st.divider()
        scan_result_panel(by_name)
        manual_add_panel(by_name)

    with right:
        cart_panel(by_name)

    return live_job


def live_camera_controls(detector: Detector, settings: Settings) -> Callable[[], None] | None:
    c1, c2 = st.columns(2)
    running = c1.toggle("Kamerayı aç", key="cam_on")
    if c2.button("🔍 Sepeti Tara", type="primary", disabled=not running, width="stretch"):
        buffer: ScanBuffer = st.session_state.scan_buffer
        if len(buffer) == 0:
            st.warning("Henüz kare alınmadı, kamera açıldıktan sonra birkaç saniye bekleyin.")
        else:
            result = buffer.counts(tracked=settings.use_tracking)
            st.session_state.scan_result = result
            if not result:
                st.warning("Son karelerde katalogdaki bir ürün görülmedi.")

    frame_ph = st.empty()
    info_ph = st.empty()
    if not running:
        release_camera()
        frame_ph.info("Canlı görüntü için **Kamerayı aç** anahtarını açın. "
                      f"Sayım, son {config.SCAN_WINDOW} kare üzerinden yapılır.")
        return None

    try:
        cap = acquire_camera(settings.camera_index)
    except CameraError as exc:
        frame_ph.error(str(exc))
        return None

    def loop() -> None:
        run_live_loop(cap, detector, settings, frame_ph, info_ph)

    return loop


def run_live_loop(cap: cv2.VideoCapture, detector: Detector, settings: Settings, frame_ph, info_ph) -> None:
    """Kameradan kare okuyup tespit eder ve ekrana basar.

    Streamlit herhangi bir butona basıldığında betiği yeniden başlatır; bu döngü
    sayfanın EN SONUNDA çalıştığı için diğer bileşenler önce çizilmiş olur.
    """
    buffer: ScanBuffer = st.session_state.scan_buffer
    failures = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            failures += 1
            if failures > 30:
                release_camera()
                frame_ph.error("Kameradan görüntü alınamıyor. Kamera bağlantısını kontrol edip anahtarı kapatıp açın.")
                return
            time.sleep(0.05)
            continue
        failures = 0
        t0 = time.perf_counter()
        detections: list[Detection] = detector.detect(frame, settings.conf, settings.iou, track=settings.use_tracking)
        buffer.add(detections)
        shown = cv2.cvtColor(annotate(frame, detections), cv2.COLOR_BGR2RGB)
        fps = 1.0 / max(time.perf_counter() - t0, 1e-6)
        frame_ph.image(shown, channels="RGB", output_format="JPEG", width="stretch")
        counts = count_detections(detections)
        summary = ", ".join(f"{by_tr(d, detections)} ×{n}" for d, n in counts.items()) or "ürün yok"
        info_ph.caption(f"⚡ {fps:.1f} FPS · Bu karede: {summary} · Tampon: {len(buffer)}/{buffer.window} kare")


def by_tr(oiv7_name: str, detections: list[Detection]) -> str:
    """Tespit listesinden ürünün Türkçe adını bulur."""
    return next((d.tr_name for d in detections if d.oiv7_name == oiv7_name), oiv7_name)


def photo_input(detector: Detector, settings: Settings, mode: str) -> None:
    if mode == MODE_SNAPSHOT:
        file = st.camera_input("Ürünlerin fotoğrafını çekin")
    else:
        file = st.file_uploader("Ürün fotoğrafı yükleyin", type=["jpg", "jpeg", "png", "bmp", "webp"])
    if file is None:
        st.caption("Ürünleri masaya yayın; üst üste binmemeleri tanımayı kolaylaştırır.")
        return

    data = file.getvalue()
    try:
        annotated, counts = detect_image(detector, settings.model_name, data, settings.conf, settings.iou)
    except ValueError as exc:
        st.error(str(exc))
        return
    st.image(annotated, width="stretch")

    # Yeni fotoğraf (veya yeni ayar) geldiyse tarama sonucunu güncelle
    signature = (hashlib.md5(data).hexdigest(), settings.conf, settings.iou, settings.model_name)
    if st.session_state.get("_photo_sig") != signature:
        st.session_state["_photo_sig"] = signature
        st.session_state.scan_result = counts
        if not counts:
            st.warning("Fotoğrafta katalogdaki bir ürün bulunamadı. Güven eşiğini düşürmeyi deneyin.")


def scan_result_panel(by_name: dict[str, Product]) -> None:
    result: dict[str, int] = st.session_state.scan_result
    st.markdown("#### 🔍 Tarama sonucu")
    if not result:
        st.caption("Henüz tarama yok. Canlı modda **Sepeti Tara**'ya basın ya da fotoğraf yükleyin.")
        return
    rows = [
        {"Ürün": by_name[n].tr_name, "Adet": q, "Birim fiyat": format_tl(by_name[n].price),
         "Tutar": format_tl(by_name[n].price * q)}
        for n, q in result.items() if n in by_name
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    c1, c2, c3 = st.columns(3)
    c1.button("➕ Sepete Ekle", type="primary", on_click=add_scan_to_cart, width="stretch")
    c2.button("🔁 Sepeti bununla değiştir", on_click=replace_cart_with_scan, width="stretch")
    c3.button("✖ Sonucu at", on_click=discard_scan, width="stretch")


def manual_add_panel(by_name: dict[str, Product]) -> None:
    with st.expander("✍️ Elle ürün ekle (model tanımazsa)"):
        names = sorted(by_name, key=lambda n: by_name[n].tr_name)
        c1, c2, c3 = st.columns([4, 1, 2], vertical_alignment="bottom")
        c1.selectbox("Ürün", names, key="manual_product",
                     format_func=lambda n: f"{by_name[n].tr_name} – {format_tl(by_name[n].price)}")
        c2.number_input("Adet", 1, 99, 1, key="manual_qty")
        c3.button("Ekle", on_click=manual_add, width="stretch")


def cart_panel(by_name: dict[str, Product]) -> None:
    st.subheader("🧾 Canlı Fiş")
    cart: dict[str, int] = st.session_state.cart
    if not cart:
        st.info("Sepet boş. Ürünleri tarayıp **Sepete Ekle**'ye basın.")
        return

    for name, qty in list(cart.items()):
        product = by_name.get(name)
        if product is None:  # katalogdan silinmiş ürün
            cart.pop(name, None)
            continue
        c_name, c_minus, c_plus, c_del = st.columns([6, 1, 1, 1], vertical_alignment="center")
        c_name.markdown(
            f"**{product.tr_name}**  \n{qty} × {format_tl(product.price)} = **{format_tl(product.price * qty)}**"
        )
        c_minus.button("➖", key=f"dec_{name}", on_click=change_qty, args=(name, -1), help="Adet azalt")
        c_plus.button("➕", key=f"inc_{name}", on_click=change_qty, args=(name, 1), help="Adet artır")
        c_del.button("🗑️", key=f"del_{name}", on_click=remove_line, args=(name,), help="Satırı sil")

    receipt = current_receipt(by_name)
    st.metric("Genel toplam", format_tl(receipt.total))
    st.code(render_text(receipt), language="text")

    c1, c2 = st.columns(2)
    c1.download_button("📄 PDF indir", data=render_pdf(receipt), file_name=f"fis_{receipt.receipt_no}.pdf",
                       mime="application/pdf", width="stretch")
    c2.download_button("🖼️ PNG kaydet", data=render_png(receipt), file_name=f"fis_{receipt.receipt_no}.png",
                       mime="image/png", width="stretch")
    c3, c4 = st.columns(2)
    c3.button("🧹 Sepeti Temizle", on_click=clear_cart, width="stretch")
    c4.button("💳 Ödemeye Geç", type="primary", on_click=go_to, args=(PAGE_PAYMENT,), width="stretch")


# ---------------------------------------------------------------------------
# Ödeme sayfası (DEMO)
# ---------------------------------------------------------------------------
SUCCESS_HTML = """
<style>
@keyframes pop {{ 0% {{transform: scale(0); opacity: 0;}} 60% {{transform: scale(1.15); opacity: 1;}}
                 100% {{transform: scale(1);}} }}
@keyframes fadeup {{ from {{opacity: 0; transform: translateY(12px);}} to {{opacity: 1; transform: none;}} }}
.pay-ok {{ text-align: center; padding: 2rem 1rem; }}
.pay-ok .circle {{ width: 120px; height: 120px; margin: 0 auto; border-radius: 50%; background: #16a34a;
                  color: #fff; font-size: 72px; line-height: 120px; animation: pop .6s ease-out both; }}
.pay-ok h2 {{ color: #16a34a; margin-top: 1rem; animation: fadeup .5s .4s ease-out both; }}
.pay-ok p {{ animation: fadeup .5s .6s ease-out both; }}
</style>
<div class="pay-ok">
  <div class="circle">✓</div>
  <h2>Ödeme Alındı ✓</h2>
  <p><b>{total}</b> · Fiş No: {receipt_no}</p>
  <p style="color:#b91c1c">DEMO – gerçek ödeme alınmamıştır.</p>
</div>
"""


def finalize_sale(receipt: Receipt) -> None:
    """Satışı kaydeder (sales.csv + fisler/ klasörü) ve sepeti sıfırlar."""
    errors: list[str] = []
    try:
        append_sale_to_csv(receipt, config.SALES_CSV)
    except OSError as exc:
        errors.append(f"sales.csv yazılamadı: {exc}")
    files = None
    try:
        files = save_receipt_files(receipt, config.RECEIPTS_DIR)
    except OSError as exc:
        errors.append(f"Fiş dosyaları kaydedilemedi: {exc}")

    ss = st.session_state
    ss.last_sale = CompletedSale(receipt=receipt, pdf=render_pdf(receipt), png=render_png(receipt),
                                 files=files, errors=errors)
    ss.cart = {}
    ss.scan_result = {}
    ss.scan_buffer.clear()
    ss.receipt_no = generate_receipt_no()
    ss.payment_done = True
    ss["_celebrate"] = True


def page_payment(by_name: dict[str, Product]) -> None:
    st.header("💳 Ödeme (DEMO)")
    st.warning("Bu ekran yalnızca gösterim amaçlıdır. **Gerçek ödeme alınmaz**, hiçbir ödeme sistemine bağlı değildir.")
    ss = st.session_state

    sale: CompletedSale | None = ss.last_sale
    if ss.payment_done and sale is not None:
        st.markdown(SUCCESS_HTML.format(total=format_tl(sale.receipt.total), receipt_no=sale.receipt.receipt_no),
                    unsafe_allow_html=True)
        if ss.pop("_celebrate", False):
            st.balloons()
        for err in sale.errors:
            st.error(err)
        if sale.files:
            st.caption(f"Fiş kaydedildi: `{sale.files[0].name}` ve `{sale.files[1].name}` → `fisler/` klasörü")
        c1, c2, c3 = st.columns(3)
        c1.download_button("📄 Fişi PDF indir", data=sale.pdf, file_name=f"fis_{sale.receipt.receipt_no}.pdf",
                           mime="application/pdf", width="stretch")
        c2.download_button("🖼️ Fişi PNG kaydet", data=sale.png, file_name=f"fis_{sale.receipt.receipt_no}.png",
                           mime="image/png", width="stretch")
        c3.button("🛒 Yeni müşteri", type="primary", on_click=new_customer, width="stretch")
        return

    if not ss.cart:
        st.info("Sepet boş. Önce kasa ekranında ürün ekleyin.")
        st.button("⬅️ Kasaya dön", on_click=go_to, args=(PAGE_CHECKOUT,))
        return

    receipt = current_receipt(by_name)
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.metric("Ödenecek tutar", format_tl(receipt.total))
        st.code(render_text(receipt), language="text")
    with right:
        st.image(make_demo_qr(receipt), caption="DEMO – gerçek ödeme değildir", width="stretch")
        st.button("⬅️ Sepete dön", on_click=go_to, args=(PAGE_CHECKOUT,), width="stretch")
        if st.button("✅ Ödemeyi Onayla (DEMO)", type="primary", width="stretch"):
            with st.spinner("Ödeme işleniyor… (demo)"):
                time.sleep(1.2)
            finalize_sale(receipt)
            st.rerun()


# ---------------------------------------------------------------------------
# Fiyat düzenleme sayfası
# ---------------------------------------------------------------------------
def page_prices(products: list[Product]) -> None:
    st.header("🏷️ Fiyat ve KDV Düzenleme")
    st.caption("Fiyatlar KDV dahil adet fiyatıdır. Değişiklikler **Kaydet**'e basınca products.json'a yazılır.")

    df = pd.DataFrame([p.to_dict() for p in products])
    kdv_options = sorted(set(config.KDV_RATES) | set(df["kdv"].astype(int)))
    edited = st.data_editor(
        df,
        key=f"price_editor_{st.session_state.get('_price_editor_ver', 0)}",
        hide_index=True,
        num_rows="fixed",
        width="stretch",
        height=600,
        column_order=["tr_name", "category", "price", "kdv", "oiv7_name"],
        column_config={
            "tr_name": st.column_config.TextColumn("Ürün", disabled=True),
            "category": st.column_config.TextColumn("Kategori", disabled=True),
            "price": st.column_config.NumberColumn("Fiyat (₺, KDV dahil)", min_value=0.0, step=0.25,
                                                   format="%.2f", required=True),
            "kdv": st.column_config.SelectboxColumn("KDV %", options=kdv_options, required=True),
            "oiv7_name": st.column_config.TextColumn("Model sınıfı (OIV7)", disabled=True),
        },
    )

    changed = (edited["price"].astype(float) != df["price"].astype(float)) | (
        edited["kdv"].astype(int) != df["kdv"].astype(int))
    st.caption(f"Değişen ürün sayısı: **{int(changed.sum())}**")

    c1, c2 = st.columns([1, 4])
    if c1.button("💾 Kaydet", type="primary", disabled=not changed.any()):
        try:
            new_products = [
                Product(oiv7_name=str(r.oiv7_name), tr_name=str(r.tr_name), price=float(r.price),
                        kdv=int(r.kdv), category=str(r.category))
                for r in edited.itertuples(index=False)
            ]
            save_catalog(new_products, config.PRODUCTS_JSON)
        except (CatalogError, ValueError, TypeError, OSError) as exc:
            st.error(f"Kaydedilemedi: {exc}")
        else:
            st.session_state["_price_editor_ver"] = st.session_state.get("_price_editor_ver", 0) + 1
            st.session_state.flash = ("success", f"{int(changed.sum())} ürünün fiyat/KDV bilgisi kaydedildi.")
            st.rerun()
    if c2.button("↩️ Değişiklikleri geri al", disabled=not changed.any()):
        st.session_state["_price_editor_ver"] = st.session_state.get("_price_editor_ver", 0) + 1
        st.rerun()


# ---------------------------------------------------------------------------
# Satış geçmişi sayfası
# ---------------------------------------------------------------------------
def load_sales() -> pd.DataFrame:
    if not config.SALES_CSV.exists() or config.SALES_CSV.stat().st_size == 0:
        return pd.DataFrame()
    df = pd.read_csv(config.SALES_CSV, encoding="utf-8-sig")
    df["tarih"] = pd.to_datetime(df["tarih"])
    df["gun"] = df["tarih"].dt.normalize()
    return df


def page_history() -> None:
    st.header("📈 Satış Geçmişi")
    try:
        df = load_sales()
    except Exception as exc:  # bozuk CSV
        st.error(f"sales.csv okunamadı: {exc}")
        return
    if df.empty:
        st.info("Henüz tamamlanmış satış yok. Ödeme ekranında bir satışı onayladığınızda burada görünür.")
        return

    first, last = df["gun"].min().date(), df["gun"].max().date()
    default_start = max(first, last - timedelta(days=29))
    picked = st.date_input("Tarih aralığı", value=(default_start, last), min_value=first, max_value=last,
                           format="DD.MM.YYYY")
    start, end = (picked if isinstance(picked, tuple) and len(picked) == 2 else (picked, picked))
    if not isinstance(start, date):
        start, end = first, last
    mask = (df["gun"].dt.date >= start) & (df["gun"].dt.date <= end)
    view = df[mask]
    if view.empty:
        st.info("Seçilen aralıkta satış yok.")
        return

    receipts = view["fis_no"].nunique()
    revenue = float(view["tutar"].sum())
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Toplam ciro", format_tl(revenue))
    m2.metric("Satış (fiş) sayısı", f"{receipts}")
    m3.metric("Satılan ürün adedi", f"{int(view['adet'].sum())}")
    m4.metric("Ortalama sepet", format_tl(revenue / receipts if receipts else 0))

    daily = view.groupby("gun", as_index=False)["tutar"].sum()
    daily["ciro_tl"] = daily["tutar"].map(format_tl)
    st.subheader("Günlük ciro")
    daily_chart = (
        alt.Chart(daily)
        .mark_bar(color="#2563eb", cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("gun:T", title="Gün", axis=alt.Axis(format="%d.%m")),
            y=alt.Y("tutar:Q", title="Ciro (₺)"),
            tooltip=[alt.Tooltip("gun:T", title="Gün", format="%d.%m.%Y"), alt.Tooltip("ciro_tl:N", title="Ciro")],
        )
        .properties(height=320)
    )
    st.altair_chart(daily_chart, width="stretch")

    st.subheader("En çok satan ürünler")
    by_qty = st.radio("Sıralama", ["Adede göre", "Ciroya göre"], horizontal=True)
    metric = "adet" if by_qty == "Adede göre" else "tutar"
    top = view.groupby("urun", as_index=False)[["adet", "tutar"]].sum().nlargest(10, metric)
    top["ciro_tl"] = top["tutar"].map(format_tl)
    top_chart = (
        alt.Chart(top)
        .mark_bar(color="#16a34a", cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
        .encode(
            x=alt.X(f"{metric}:Q", title="Adet" if metric == "adet" else "Ciro (₺)"),
            y=alt.Y("urun:N", sort="-x", title=None),
            tooltip=[alt.Tooltip("urun:N", title="Ürün"), alt.Tooltip("adet:Q", title="Adet"),
                     alt.Tooltip("ciro_tl:N", title="Ciro")],
        )
        .properties(height=max(200, 32 * len(top)))
    )
    st.altair_chart(top_chart, width="stretch")

    with st.expander("Satış kayıtları"):
        table = view.sort_values("tarih", ascending=False)[
            ["tarih", "fis_no", "urun", "adet", "birim_fiyat", "kdv_orani", "tutar"]]
        st.dataframe(table, hide_index=True, width="stretch")
        st.download_button("⬇️ sales.csv indir", data=config.SALES_CSV.read_bytes(), file_name="sales.csv",
                           mime="text/csv")


# ---------------------------------------------------------------------------
# Ana akış
# ---------------------------------------------------------------------------
def main() -> None:
    init_state()
    settings = sidebar()
    products = get_products()
    by_name = products_by_name(products)

    flash = st.session_state.flash
    if flash:
        kind, message = flash
        getattr(st, kind)(message)
        st.session_state.flash = None

    page = st.session_state.page
    live_job: Callable[[], None] | None = None
    if page == PAGE_CHECKOUT:
        st.title(f"🛒 {config.APP_NAME}")
        live_job = page_checkout(settings, by_name)
    else:
        release_camera()  # başka sayfadayken kamerayı serbest bırak
        if page == PAGE_PAYMENT:
            page_payment(by_name)
        elif page == PAGE_PRICES:
            page_prices(products)
        elif page == PAGE_HISTORY:
            page_history()

    # Canlı kamera döngüsü her zaman en son çalışır (sonsuz döngü).
    if live_job is not None:
        live_job()


main()
