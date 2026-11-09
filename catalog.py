"""Ürün kataloğu: products.json okuma/yazma ve model sınıf adlarıyla eşleştirme."""
from __future__ import annotations

import difflib
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import config


class CatalogError(Exception):
    """Katalog dosyası okunamadığında veya geçersiz olduğunda fırlatılır."""


@dataclass
class Product:
    """Katalogdaki tek bir ürün."""

    oiv7_name: str  # Open Images V7 sınıf adı (modelin döndürdüğü ad)
    tr_name: str  # Fişte görünen Türkçe ad
    price: float  # KDV dahil adet fiyatı (TL)
    kdv: int  # KDV oranı (%)
    category: str  # Kategori (Meyve, Sebze, ...)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class MatchResult:
    """Katalog ile model sınıflarının eşleştirme sonucu."""

    class_map: dict[int, Product] = field(default_factory=dict)  # model sınıf id -> ürün
    missing: list[tuple[str, list[str]]] = field(default_factory=list)  # (ad, öneriler)


REQUIRED_FIELDS: tuple[str, ...] = ("oiv7_name", "tr_name", "price", "kdv", "category")


def _validate(raw: Mapping[str, object], index: int) -> Product:
    """Tek bir JSON kaydını doğrulayıp Product nesnesine çevirir."""
    eksik = [f for f in REQUIRED_FIELDS if f not in raw]
    if eksik:
        raise CatalogError(f"{index + 1}. üründe eksik alan(lar): {', '.join(eksik)}")
    try:
        price = float(raw["price"])  # type: ignore[arg-type]
        kdv = int(raw["kdv"])  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise CatalogError(f"{index + 1}. üründe fiyat/KDV sayı değil: {raw}") from exc
    if price != price or price < 0:  # NaN veya negatif
        raise CatalogError(f"'{raw['tr_name']}' için fiyat geçersiz (negatif veya boş olamaz).")
    if not 0 <= kdv <= 100:
        raise CatalogError(f"'{raw['tr_name']}' için KDV oranı 0-100 arasında olmalı.")
    return Product(
        oiv7_name=str(raw["oiv7_name"]).strip(),
        tr_name=str(raw["tr_name"]).strip(),
        price=price,
        kdv=kdv,
        category=str(raw["category"]).strip(),
    )


def load_catalog(path: Path | str = config.PRODUCTS_JSON) -> list[Product]:
    """products.json dosyasını okur ve ürün listesini döndürür."""
    path = Path(path)
    if not path.is_file():
        raise CatalogError(f"Ürün kataloğu bulunamadı: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CatalogError(f"products.json geçerli bir JSON değil: {exc}") from exc
    if not isinstance(data, list):
        raise CatalogError("products.json bir liste ([...]) içermeli.")

    products = [_validate(item, i) for i, item in enumerate(data)]
    # Aynı oiv7_name iki kez tanımlanmasın
    seen: set[str] = set()
    for p in products:
        key = p.oiv7_name.lower()
        if key in seen:
            raise CatalogError(f"'{p.oiv7_name}' katalogda birden fazla kez geçiyor.")
        seen.add(key)
    return products


def save_catalog(products: Iterable[Product], path: Path | str = config.PRODUCTS_JSON) -> None:
    """Ürün listesini doğrulayıp products.json'a yazar (önce geçici dosyaya, sonra yer değiştirir)."""
    path = Path(path)
    items = [_validate(p.to_dict(), i).to_dict() for i, p in enumerate(products)]
    # Tam sayı fiyatları JSON'da 12.0 yerine 12 olarak sakla
    for item in items:
        price = float(item["price"])  # type: ignore[arg-type]
        item["price"] = int(price) if price.is_integer() else round(price, 2)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(items, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def products_by_name(products: Iterable[Product]) -> dict[str, Product]:
    """oiv7_name -> Product sözlüğü."""
    return {p.oiv7_name: p for p in products}


def match_catalog(
    model_names: Mapping[int, str] | Sequence[str],
    products: Iterable[Product],
    warn: bool = True,
) -> MatchResult:
    """Katalogdaki oiv7_name değerlerini modelin sınıf adlarıyla eşleştirir.

    Eşleştirme büyük/küçük harf duyarsızdır. Modelde bulunamayan adlar için
    difflib ile en yakın sınıf adları önerilir ve (warn=True ise) konsola yazdırılır.
    """
    names: dict[int, str] = (
        dict(model_names) if isinstance(model_names, Mapping) else dict(enumerate(model_names))
    )
    lower_to_id: dict[str, int] = {n.strip().lower(): i for i, n in names.items()}
    result = MatchResult()

    for product in products:
        key = product.oiv7_name.strip().lower()
        cls_id = lower_to_id.get(key)
        if cls_id is not None:
            result.class_map[cls_id] = product
            continue
        close = difflib.get_close_matches(key, list(lower_to_id), n=3, cutoff=0.55)
        suggestions = [names[lower_to_id[c]] for c in close]
        result.missing.append((product.oiv7_name, suggestions))

    if warn and result.missing:
        for name, suggestions in result.missing:
            hint = f" En yakın sınıf(lar): {', '.join(suggestions)}" if suggestions else " Benzer sınıf yok."
            print(f"[UYARI] '{name}' modelin sınıfları arasında bulunamadı, yok sayılacak.{hint}", file=sys.stderr)
        print(
            f"[BİLGİ] {len(result.class_map)} ürün eşleşti, {len(result.missing)} ürün eşleşmedi.",
            file=sys.stderr,
        )
    return result
