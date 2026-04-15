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


def test_item_slots_are_patch_aggregate_not_match_build(fixture_match_id):
    # The item-stat region (offsets 100..119) must be identical across any two
    # matches on the same patch — it's a patch-level aggregate, not a per-match
    # end-of-game build. Reading final builds would leak outcomes into the
    # static prefix token.
    from db import get_conn
    conn = get_conn()
    rows = conn.execute(
        "SELECT match_id FROM games WHERE patch = (SELECT patch FROM games WHERE match_id = ?) LIMIT 3",
        (fixture_match_id,),
    ).fetchall()
    conn.close()
    ids = [r["match_id"] for r in rows]
    assert len(ids) >= 2, "need at least two matches on the same patch"
    vecs = [patch_vector_for_match(mid) for mid in ids]
    for other in vecs[1:]:
        np.testing.assert_allclose(vecs[0][100:120], other[100:120])
