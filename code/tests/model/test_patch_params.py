import numpy as np
from model.patch_params import (
    fetch_dragon_version, load_items, load_champions,
    patch_vector_for_match, PATCH_VECTOR_DIM,
)


def test_dragon_version_format():
    v = fetch_dragon_version("14.14.1")
    assert v == "14.14.1"


def test_load_items_returns_dict():
    items = load_items("14.14.1")
    # At least a few well-known items in 14.x.
    assert any("BF Sword" in i.get("name", "") for i in items.values()) or \
           any("B. F. Sword" in i.get("name", "") for i in items.values())


def test_load_champions_returns_dict():
    champs = load_champions("14.14.1")
    # 160+ champs by late season 14.
    assert len(champs) > 150


def test_patch_vector_shape(fixture_match_id):
    vec = patch_vector_for_match(fixture_match_id)
    assert vec.shape == (PATCH_VECTOR_DIM,)
    assert vec.dtype == np.float32


def test_patch_vector_is_finite(fixture_match_id):
    vec = patch_vector_for_match(fixture_match_id)
    assert np.all(np.isfinite(vec))
