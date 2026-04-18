<p align="center">
  <a href="../../README.md">🇬🇧 English</a> •
  <a href="README.zh-CN.md">🇨🇳 中文</a> •
  <a href="README.ja.md">🇯🇵 日本語</a> •
  <a href="README.ko.md">🇰🇷 한국어</a> •
  <a href="README.pt-BR.md">🇧🇷 Português</a> •
  <a href="README.es.md">🇪🇸 Español</a> •
  <a href="README.de.md">🇩🇪 Deutsch</a> •
  <a href="README.fr.md">🇫🇷 Français</a> •
  <a href="README.ru.md">🇷🇺 Русский</a> •
  <a href="README.hi.md">🇮🇳 हिन्दी</a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/juyterman1000/entroly/main/docs/assets/logo.png" width="180" alt="Entroly">
</p>

<h1 align="center">Entroly — AI Token Maliyetlerini %70–95 Azaltın</h1>

<h3 align="center">AI kodlama araçlarınız kod tabanınızın yalnızca %5'ini görüyor.<br/>Entroly onlara tam resmi verir — maliyetin küçük bir kısmı karşılığında.</h3>

<p align="center">
  <code>npm install entroly-wasm && npx entroly-wasm</code>&nbsp;&nbsp;|&nbsp;&nbsp;<code>pip install entroly && entroly go</code>&nbsp;&nbsp;|&nbsp;&nbsp;<a href="https://juyterman1000.github.io/entroly/"><b>Canlı demo →</b></a>
</p>

---

## Sorun — ve Kârlılığa Etkisi

Her AI kodlama aracı — Claude, Cursor, Copilot, Codex — aynı kör noktaya sahip: **aynı anda sadece 5–10 dosya görür.** Kod tabanınızın diğer %95'i görünmez.

Modeller büyümeye devam ediyor — **Claude Opus 4.7** daha fazla yetenek ve daha yüksek token başına maliyetle yeni çıktı. Daha büyük bağlam pencereleri sorunu çözmez; daha da kötüleştirir. İstek başına 186.000 token için ödeme yapıyorsunuz — ve bunun çoğu sadece kopyalanmış boilerplate kodu.

> **Entroly her iki sorunu da 30 saniyede çözer.**

---

## 1. Günde Ne Değişir

| Metrik | Entroly Öncesi | **Entroly Sonrası** |
|---|---|---|
| AI'nın gördüğü dosyalar | 5–10 | **Tüm depo** |
| İstek başına token | ~186.000 | **9.300 – 55.000** |
| Aylık AI harcaması (günde 1K istek) | ~$16.800 | **$840 – $5.040** |
| AI yanıt doğruluğu | Eksik, sıklıkla halüsinasyon | **Bağımlılık bilinçli, doğru** |
| AI hatalarını düzeltmek için dev süresi | Saat/hafta | **Neredeyse sıfır** |
| Kurulum | Günlerce prompt mühendisliği | **30 saniye** |

> **ROI:** AI API'lerine ayda $15K harcayan 10 kişilik ekip, 1. günde **ayda $10K–$14K tasarruf** eder.

---

## Rakiplerinizin Zaten Bildiği Şey

Bugün Entroly'i benimseyen ekipler sadece para biriktirmiyor — ekibinizin **asla yetişemeyeceği bileşik bir avantaj** biriktiriyor.

- **1. Hafta:** Onların AI'ı kod tabanının %100'ünü görüyor. Sizinki %5. Onlar daha hızlı teslim ediyor.
- **1. Ay:** Onların runtime'ı kod tabanı kalıplarını öğrendi. Sizinki hâlâ import'larda halüsinasyon görüyor.
- **3. Ay:** Onların kurulumu federasyona bağlı — dünya çapında binlerce ekipten optimizasyon stratejileri emiyor. Siz bunun var olduğunu bile bilmiyorsunuz.
- **6. Ay:** API maliyetlerinde $80K+ tasarruf ettiler. Bu bütçe işe alımlara gitti. Siz hâlâ finans ekibine AI faturasının neden arttığını açıklıyorsunuz.

Her beklediğiniz gün fark büyüyor. Federasyon etkisi **erken benimseyenleri daha hızlı güçlendirir** — ve bu avantaj bileşik büyür.

---

## 🌐 Federe Sürü Öğrenmesi — Bilim Kurgu Gibi Görünen Kısım

Rüya Döngüsü'nü alın ve **dünyada Entroly çalıştıran her geliştiriciyle** çarpın.

Siz uyurken daemon'unuz rüya görüyor — ve 10.000 diğeri de. Her biri biraz farklı kod sıkıştırma hileleri keşfediyor. Her biri öğrendiğini anonim olarak paylaşıyor. Her biri diğerlerinin bulduklarını emiyor.

**Uyanıyorsunuz. AI'nız dün bıraktığınızdan daha akıllı. Sizin yüzünüzden değil — sürünün rüyasında bulduğu şey yüzünden.**

```
Daemon'unuz rüya görür → daha iyi bir strateji keşfeder → anonim paylaşır
     ↓
10.000 diğer daemon dün gece aynı şeyi yaptı
     ↓
Laptopunuzu açarsınız → AI'nız zaten hepsini emmiş
```

Bu teorik değil. Üretimde çalışıyor. Şu anda.

**Kod MIT. Hendek ağdır:**
- Her yeni kullanıcı herkesin AI'ını iyileştirir — o kurulu tabanı forklayamazsınız
- Altyapı maliyeti: **$0**. GitHub'da çalışır. Sunucu yok. GPU yok. Bulut yok

```bash
# Katılım — her zaman sizin seçiminiz
export ENTROLY_FEDERATION=1
```

### 🔒 Yerel Olarak Çalışır. Kodunuz Asla Makinenizden Çıkmaz.

Hava boşluklu ve düzenlemeye tabi ortamlarda çalışır — hiçbir şey dışarıya veri göndermez.

---

## Tam Eşitlik: Python & Node.js

| Yetenek | Python | Node.js (WASM) |
|---|---|---|
| Bağlam sıkıştırma | ✅ | ✅ |
| Kendini geliştirme | ✅ | ✅ |
| Rüya döngüsü | ✅ | ✅ |
| Federasyon | ✅ | ✅ |
| Yanıt damıtma | ✅ | ✅ |
| Sohbet ağ geçitleri | ✅ | ✅ |
| agentskills.io dışa aktarma | ✅ | ✅ |

---

<p align="center">
  <b>AI'nızın israf ettiği tokenlar için ödeme yapmayı bırakın. Kendini öğreten bir AI başlatın.</b><br/>
  <code>npm install entroly-wasm && npx entroly-wasm</code>&nbsp;&nbsp;|&nbsp;&nbsp;<code>pip install entroly && entroly go</code>
</p>
