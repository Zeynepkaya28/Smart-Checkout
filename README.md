# 🛒 Smart Checkout (YOLO)

Masaya konan market ürünlerini kamerayla tanıyan ve kasiyersiz kasa gibi otomatik fiş çıkaran bir demo.
Ultralytics'in **Open Images V7** üzerinde eğitilmiş hazır modeli (`yolov8m-oiv7.pt`, 601 sınıf) kullanılır;
model yalnızca `products.json` içindeki **70 ürünü** tespit eder, diğer sınıfları yok sayar.

![Ekran görüntüsü](docs/ekran_goruntusu.png)
<!-- Uygulamanın ekran görüntüsünü docs/ekran_goruntusu.png olarak kaydedin. -->

## Özellikler

- **Canlı kamera** (OpenCV) – kutular ve Türkçe etiketler ekranda çizilir
- **Fotoğraf** – tarayıcıdan fotoğraf çekme (`st.camera_input`) veya dosya yükleme
- **Çift sayım önleme** – "Sepeti Tara" son 15 karede her ürün için görülen adetlerin **medyanını** alır;
  istenirse **ByteTrack** ile benzersiz takip ID'si sayılır
- Sepete ekle, sepeti temizle, satır sil, adet +/−, elle ürün ekleme
- **Canlı fiş**: tarih-saat, fiş no, ürün/adet/birim fiyat/tutar, KDV oranına göre döküm, genel toplam
  (Türkçe para formatı: `1.234,50 ₺`)
- Fişi **PDF** (DejaVuSans gömülü, Türkçe karakterler sorunsuz) ve **PNG** olarak indirme
- **Fiyat düzenleme** sayfası (`st.data_editor` ile fiyat ve KDV)
- **Demo ödeme** ekranı: QR kod + "Ödeme Alındı ✓" animasyonu – *gerçek ödeme entegrasyonu yoktur*
- **Satış geçmişi**: `sales.csv`, günlük ciro ve en çok satan ürünler grafikleri

## Proje yapısı

```
smart_checkout/
├─ app.py              Streamlit arayüzü
├─ detector.py         YOLO yükleme, sınıf filtreleme, çizim, sayım (medyan / ByteTrack)
├─ receipt.py          Fiş hesaplama (KDV), metin/PDF/PNG fiş, QR, sales.csv kaydı
├─ catalog.py          products.json okuma/yazma, model sınıf adlarıyla eşleştirme
├─ config.py           Ayarlar (model boyutu, eşikler, dosya yolları, font bulma)
├─ live_cam.py         Streamlit'siz hızlı OpenCV demosu (S=tara, C=temizle, Q=çık)
├─ train_custom.py     Marka tanıma için fine-tune iskeleti
├─ launcher.py         Başlatıcı: Streamlit sunucusunu açar + tarayıcıyı başlatır (.exe'nin giriş noktası)
├─ smart_checkout.spec    PyInstaller paketleme tarifi
├─ build_exe.ps1       Tek komutla SmartCheckout.exe derleme betiği
├─ SmartCheckout.bat      Paketlemeden çift tıklayarak çalıştırma
├─ products.json       70 ürünlük katalog
├─ requirements.txt
├─ pytest.ini
└─ tests/
   ├─ test_receipt.py  KDV ve toplam hesaplarının testleri
   └─ test_catalog.py  Katalog ve isim eşleştirme testleri
```

Çalışma sırasında oluşanlar: `sales.csv` (satışlar), `fisler/` (her satışın PDF+PNG fişi),
`yolov8*-oiv7.pt` (indirilen model).

## Kurulum (Windows + NVIDIA GPU)

> Gereksinimler: Python 3.10 – 3.13, güncel NVIDIA sürücüsü. Ayrıca CUDA Toolkit kurmanıza **gerek yok**;
> PyTorch'un CUDA'lı paketi gerekli kütüphaneleri kendisi getirir.

PowerShell'de proje klasörüne girin:

**1) Sanal ortam (venv) oluşturun ve etkinleştirin**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

> `Activate.ps1` "running scripts is disabled" hatası verirse bir kez şunu çalıştırın:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

**2) CUDA'lı PyTorch'u kurun (ÖNCE bu!)**

Sürücünüzün desteklediği CUDA sürümünü `nvidia-smi` çıktısının sağ üstünde görebilirsiniz.
<https://pytorch.org/get-started/locally/> adresindeki seçiciden (Windows / Pip / CUDA) komutu alın. Örnek:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

Kontrol edin – `True` ve ekran kartınızın adını görmelisiniz:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"
```

**3) Diğer paketleri kurun**

```powershell
pip install -r requirements.txt
```

> Sırayı değiştirip önce `requirements.txt` kurarsanız pip CPU'lu PyTorch kurar. Bu durumda
> `pip uninstall torch torchvision -y` yapıp 2. adımı tekrarlayın.

## Çalıştırma

**Streamlit uygulaması**

```powershell
streamlit run app.py
```

Tarayıcıda <http://localhost:8501> açılır. İlk çalıştırmada model (~50 MB, `m` boyutu) otomatik indirilir.

**Hızlı OpenCV demosu** (Streamlit'siz, daha yüksek FPS)

```powershell
python live_cam.py                    # varsayılan: kamera 0, model m
python live_cam.py --camera 1 --model s --conf 0.4
python live_cam.py --track            # ByteTrack ile sayım
```

Pencere seçiliyken: **S** = sepeti tara (fiş konsola yazılır), **C** = temizle, **Q** / ESC = çıkış.

**Çift tıklayarak çalıştırma (paketlemeden)**

`SmartCheckout.bat` dosyasına çift tıklayın: `.venv` içindeki Python ile `launcher.py`'yi çalıştırır,
boş bir port bulur ve tarayıcıyı kendisi açar. Konsol penceresini kapatmak uygulamayı kapatır.

```powershell
python launcher.py               # aynı şey, komut satırından
python launcher.py --port 8600   # belirli port
python launcher.py --lan         # aynı ağdaki tablet/telefondan erişim (güvenlik duvarı izni gerekir)
```

## Windows uygulaması (.exe) oluşturma

Python kurulu olmayan bir bilgisayarda çalıştırmak için uygulamayı **SmartCheckout.exe** olarak paketleyebilirsiniz.
Kurulum bölümündeki 1–3. adımlar tamamlanmış ve venv etkin olmalı:

```powershell
.\build_exe.ps1                  # dist\SmartCheckout\SmartCheckout.exe
.\build_exe.ps1 -WithModel       # modeli de pakete koyar (ilk açılışta internet gerekmez)
```

Çıktı bir **klasördür** (`dist\SmartCheckout\`): `SmartCheckout.exe` + `_internal\`. Dağıtırken klasörün tamamını
zip'leyin. Hedef bilgisayarda yalnızca `SmartCheckout.exe`'ye çift tıklamak yeterlidir; konsol penceresinde adres
görünür ve tarayıcı otomatik açılır.

- İlk açılışta `products.json` .exe'nin yanına kopyalanır; fiyat düzenlemeleri, `sales.csv`, `fisler\` ve
  indirilen model de **.exe'nin yanındaki klasöre** yazılır. Bu yüzden klasörü `C:\Program Files` gibi yazma
  izni olmayan bir yere değil, örneğin `C:\SmartCheckout` veya Masaüstü'ne koyun.
- Boyut: CUDA'lı PyTorch ile paket **~3–4 GB** olur (CPU'lu PyTorch ile ~1 GB). Tek dosya (onefile) yerine
  klasör kullanılmasının nedeni budur: onefile her açılışta GB'larca dosyayı geçici klasöre açar.
- Hedef bilgisayarda GPU kullanmak için yalnızca güncel **NVIDIA sürücüsü** gerekir; CUDA kütüphaneleri pakettedir.
- Windows SmartScreen imzasız .exe için uyarı verebilir: "Ek bilgi → Yine de çalıştır".
- Antivirüs PyInstaller paketlerini bazen yanlışlıkla işaretler; gerekirse klasörü istisnalara ekleyin.

**Testler**

```powershell
pytest
```

## Ayarlar

| Ayar | Nerede | Varsayılan |
|---|---|---|
| Güven eşiği | Kenar çubuğu | 0.35 |
| IoU eşiği | Kenar çubuğu | 0.50 |
| Model boyutu (n/s/m/l/x) | Kenar çubuğu veya `SMART_CHECKOUT_MODEL_SIZE` | m |
| Kamera indeksi | Kenar çubuğu veya `SMART_CHECKOUT_CAMERA` | 0 |
| Özel model (.pt yolu) | `SMART_CHECKOUT_MODEL` ortam değişkeni | – |
| Tarama penceresi, market adı, dosya yolları | `config.py` | 15 kare, "YOLO Market" |

Model boyutu seçimi: laptop GPU'larında `m` iyi bir dengedir. FPS düşükse `s` veya `n`, doğruluk için `l`/`x` deneyin.
CUDA yoksa uygulama otomatik olarak CPU'ya geçer (yavaş çalışır; `n` veya `s` önerilir).

### Model–katalog eşleştirme

Başlangıçta `products.json` içindeki `oiv7_name` değerleri modelin `model.names` listesiyle
**büyük/küçük harf duyarsız** eşleştirilir. Bulunamayan adlar konsola uyarı olarak yazılır ve
`difflib` ile en yakın sınıf adı önerilir, örneğin:

```
[UYARI] 'Bananna' modelin sınıfları arasında bulunamadı, yok sayılacak. En yakın sınıf(lar): Banana
```

Aynı liste uygulamada "⚠️ … katalog ürünü modelde bulunamadı" bölümünde de görünür.

### Font

PDF/PNG fiş ve kamera etiketlerinde Türkçe karakterler (ş, ğ, ı, İ) ve ₺ için **DejaVuSans** kullanılır.
Font sırasıyla `fonts/DejaVuSans.ttf`, matplotlib'in içindeki kopya (ultralytics ile gelir) ve sistem
fontlarında aranır; PDF'e gömülür. Hiçbiri bulunamazsa Arial'a, o da yoksa ASCII'ye ("TL") düşülür.

## Bilinen sınırlamalar

- **Model ürünleri türüne göre tanır, marka ayırt etmez.** "Süt", "Konserve (Tin can)", "Şişe (Bottle)"
  gibi genel sınıflar vardır; iki farklı marka süt aynı ürün sayılır ve aynı fiyattan fişe yazılır.
  Marka tanıma için ileride kendi fotoğraflarınızla **fine-tune** yapılabilir (aşağıya bakın).
- Fiyatlar **adet** fiyatıdır; tartılı satış (kg) yoktur. Meyve/sebze adetle sayılır.
- Open Images sınıfları market için tasarlanmadığından bazı ürünler (Tea, Coffee, Milk, Honeycomb, Candy)
  zor veya hatalı tanınabilir. `Bottle` her şişeyi "Su (şişe)" olarak sayar; `Mug`, `Coffee cup` vb.
  masadaki kişisel eşyaları da sayabilir – kasa alanını boş tutun.
- Üst üste yığılmış, birbirini kapatan ürünler eksik sayılabilir. Ürünleri masaya yayın.
- Streamlit canlı modu, kareleri tarayıcıya gönderdiği için `live_cam.py`'den daha düşük FPS verir.
- Ödeme ekranı tamamen **demo**dur; QR kod hiçbir ödeme sistemine bağlı değildir.
- Oluşan fişler mali belge değildir.

## Marka tanıma için fine-tune (`train_custom.py`)

1. Klasör iskeletini oluşturun:

   ```powershell
   python train_custom.py --init --classes marka_a_sut_1l marka_b_sut_1l marka_c_su_05l
   ```

   Oluşan yapı:

   ```
   datasets/marka_urunler/
   ├─ data.yaml
   ├─ images/
   │  ├─ train/   foto_0001.jpg ...
   │  ├─ val/     foto_0101.jpg ...
   │  └─ test/
   └─ labels/
      ├─ train/   foto_0001.txt ...   (her satır: sınıf_id x_merkez y_merkez genişlik yükseklik)
      ├─ val/
      └─ test/
   ```

2. Sınıf başına **100–200+** fotoğraf çekin (farklı açı, ışık, arka plan, kasadaki gerçek ortam).
   Label Studio, CVAT veya Roboflow ile YOLO formatında etiketleyin. Yaklaşık %80 train, %20 val ayırın.
3. Eğitin (GPU'da birkaç dakika–saat):

   ```powershell
   python train_custom.py --data datasets/marka_urunler/data.yaml --epochs 100
   ```

4. Uygulamada kullanın:

   ```powershell
   $env:SMART_CHECKOUT_MODEL = "runs\marka\marka_v1\weights\best.pt"
   streamlit run app.py
   ```

   `products.json` içindeki `oiv7_name` alanlarını `data.yaml`'daki sınıf adlarıyla değiştirin
   (ör. `"oiv7_name": "marka_a_sut_1l"`).

## Sorun giderme

| Belirti | Çözüm |
|---|---|
| "Kamera açılamadı" | Zoom/Teams/tarayıcı kamerayı kullanıyor olabilir; kapatın. Kamera indeksini 1 veya 2 yapın. Windows Gizlilik > Kamera izinlerini kontrol edin. |
| "Model indirilemedi" | İnternet bağlantısını kontrol edin veya `.pt` dosyasını [ultralytics/assets releases](https://github.com/ultralytics/assets/releases) sayfasından indirip proje klasörüne koyun. Yarım inmiş dosyayı silin. |
| Kenar çubuğunda "CPU" yazıyor | CUDA'lı PyTorch kurulu değil; Kurulum 2. adımı tekrarlayın. |
| ByteTrack modunda `lap` hatası | `pip install lap` |
| .exe açılıp hemen kapanıyor | Konsoldan çalıştırıp hatayı görün: `cd dist\SmartCheckout` → `.\SmartCheckout.exe`. Eksik modül hatasında paketi venv etkinken yeniden derleyin. |
| .exe'de "Model indirilemedi" | İnternet yoksa `.\build_exe.ps1 -WithModel` ile derleyin veya `.pt` dosyasını .exe'nin yanına koyun. |
| PDF'te Türkçe karakter bozuk | `fonts/` klasörüne `DejaVuSans.ttf` ve `DejaVuSans-Bold.ttf` koyun. |
