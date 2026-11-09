"""receipt.py için KDV ve toplam hesap testleri.

Çalıştırma (proje klasöründe):  pytest
"""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal as D

import pytest

from catalog import Product
from receipt import (
    CartLine,
    build_receipt,
    compute_kdv_groups,
    format_tl,
    kdv_from_gross,
    lines_from_cart,
    money,
    render_text,
)


def line(name: str, qty: int, price: str | int | float, kdv: int) -> CartLine:
    return CartLine(code=name, name=name, qty=qty, unit_price=D(str(price)), kdv_rate=kdv)


# --- Para formatı --------------------------------------------------------------
@pytest.mark.parametrize(
    ("amount", "expected"),
    [
        (D("0"), "0,00 ₺"),
        (D("5"), "5,00 ₺"),
        (D("1234.5"), "1.234,50 ₺"),
        (D("1234567.891"), "1.234.567,89 ₺"),
        (D("-12.5"), "-12,50 ₺"),
        (D("0.005"), "0,01 ₺"),  # kuruş yukarı yuvarlanır
        (1000, "1.000,00 ₺"),
        (0.1 + 0.2, "0,30 ₺"),  # float hatası görünmez
    ],
)
def test_format_tl(amount, expected):
    assert format_tl(amount) == expected


def test_format_tl_custom_symbol():
    assert format_tl(D("12.5"), "TL") == "12,50 TL"


# --- KDV hesabı ----------------------------------------------------------------
@pytest.mark.parametrize(
    ("gross", "rate", "net", "kdv"),
    [
        ("120", 20, "100.00", "20.00"),
        ("110", 10, "100.00", "10.00"),
        ("101", 1, "100.00", "1.00"),
        ("15", 1, "14.85", "0.15"),
        ("70", 20, "58.33", "11.67"),
        ("50", 0, "50.00", "0.00"),
        ("0", 20, "0.00", "0.00"),
    ],
)
def test_kdv_from_gross(gross, rate, net, kdv):
    n, k = kdv_from_gross(D(gross), rate)
    assert n == D(net)
    assert k == D(kdv)
    assert n + k == money(gross)  # matrah + KDV her zaman brüt tutara eşit


def test_line_total():
    assert line("Elma", 3, 12, 1).total == D("36.00")
    assert line("Sakız", 3, "0.1", 20).total == D("0.30")


@pytest.mark.parametrize("qty", [0, -1])
def test_invalid_qty_raises(qty):
    with pytest.raises(ValueError):
        line("Elma", qty, 12, 1)


def test_negative_price_raises():
    with pytest.raises(ValueError):
        line("Elma", 1, -5, 1)


def test_mixed_rates_receipt():
    lines = [
        line("Elma", 3, 12, 1),       # 36
        line("Bira", 2, 70, 20),      # 140
        line("Sandviç", 1, 70, 10),   # 70
        line("Muz", 2, 15, 1),        # 30
    ]
    r = build_receipt(lines, receipt_no="TEST-1", issued_at=datetime(2026, 11, 9, 15, 30))

    assert r.total == D("276.00")
    assert r.item_count == 8

    groups = {g.rate: g for g in r.kdv_groups}
    assert list(groups) == [1, 10, 20]  # küçükten büyüğe sıralı
    assert groups[1].gross == D("66.00")
    assert (groups[1].net, groups[1].kdv) == (D("65.35"), D("0.65"))
    assert groups[10].gross == D("70.00")
    assert (groups[10].net, groups[10].kdv) == (D("63.64"), D("6.36"))
    assert groups[20].gross == D("140.00")
    assert (groups[20].net, groups[20].kdv) == (D("116.67"), D("23.33"))

    assert r.net_total == D("245.66")
    assert r.kdv_total == D("30.34")
    assert r.net_total + r.kdv_total == r.total


def test_group_gross_sums_to_total():
    lines = [line(f"Ürün{i}", i % 4 + 1, D("7.99") * (i + 1), (0, 1, 10, 20)[i % 4]) for i in range(20)]
    r = build_receipt(lines)
    assert sum(g.gross for g in r.kdv_groups) == r.total
    for g in r.kdv_groups:
        assert g.net + g.kdv == g.gross


def test_empty_receipt():
    r = build_receipt([])
    assert r.total == D("0.00")
    assert r.kdv_groups == ()
    assert r.item_count == 0
    assert compute_kdv_groups([]) == []


def test_lines_from_cart_uses_catalog_prices():
    products = {
        "Apple": Product("Apple", "Elma", 12, 1, "Meyve"),
        "Beer": Product("Beer", "Bira", 70.5, 20, "İçecek"),
    }
    cart = {"Apple": 2, "Beer": 1, "Unknown": 5, "Banana": 0}
    lines = lines_from_cart(cart, products)
    assert [(l.name, l.qty, l.unit_price, l.kdv_rate) for l in lines] == [
        ("Elma", 2, D("12.00"), 1),
        ("Bira", 1, D("70.50"), 20),
    ]
    assert build_receipt(lines).total == D("94.50")


def test_render_text_contains_key_fields():
    r = build_receipt([line("Elma", 3, 12, 1), line("Bira", 1, 70, 20)], receipt_no="YM-TEST")
    text = render_text(r)
    assert "YOLO Market" in text
    assert "Fiş No:" in text and "YM-TEST" in text
    assert "GENEL TOPLAM" in text
    assert "106,00 ₺" in text
    assert "KDV DÖKÜMÜ" in text
    assert "%20" in text and "%1" in text
