"""catalog.py için isim eşleştirme ve dosya okuma/yazma testleri."""
from __future__ import annotations

import json

import pytest

import config
from catalog import CatalogError, Product, load_catalog, match_catalog, save_catalog


def test_match_is_case_insensitive_and_suggests():
    model_names = {0: "Apple", 1: "Banana", 2: "Tin can", 3: "Person"}
    products = [
        Product("apple", "Elma", 12, 1, "Meyve"),
        Product("TIN CAN", "Konserve", 55, 1, "Temel gıda"),
        Product("Bananna", "Muz", 15, 1, "Meyve"),  # yazım hatası
    ]
    result = match_catalog(model_names, products, warn=False)
    assert set(result.class_map) == {0, 2}
    assert result.class_map[2].tr_name == "Konserve"
    assert result.missing == [("Bananna", ["Banana"])]


def test_match_accepts_list_of_names():
    result = match_catalog(["Milk", "Bread"], [Product("bread", "Ekmek", 15, 1, "Fırın")], warn=False)
    assert list(result.class_map) == [1]


def test_bundled_catalog_is_valid():
    products = load_catalog(config.PRODUCTS_JSON)
    assert len(products) == 70
    assert len({p.oiv7_name.lower() for p in products}) == 70
    assert all(p.price > 0 for p in products)


def test_save_and_load_roundtrip(tmp_path):
    path = tmp_path / "products.json"
    save_catalog([Product("Apple", "Elma", 12.0, 1, "Meyve"), Product("Beer", "Bira", 70.5, 20, "İçecek")], path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw[0]["price"] == 12  # tam sayı olarak saklanır
    assert raw[1]["tr_name"] == "Bira"
    assert [p.price for p in load_catalog(path)] == [12.0, 70.5]


def test_invalid_price_rejected(tmp_path):
    path = tmp_path / "products.json"
    path.write_text(json.dumps([{"oiv7_name": "Apple", "tr_name": "Elma", "price": -1, "kdv": 1,
                                 "category": "Meyve"}]), encoding="utf-8")
    with pytest.raises(CatalogError):
        load_catalog(path)
