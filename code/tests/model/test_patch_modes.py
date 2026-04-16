from model.patch_modes import (
    PATCH_VECTOR_MODE_SCAFFOLDING, PATCH_VECTOR_MODE_FULL,
    DEFAULT_PATCH_VECTOR_MODE,
)


def test_modes_are_distinct_strings():
    assert PATCH_VECTOR_MODE_SCAFFOLDING == "scaffolding"
    assert PATCH_VECTOR_MODE_FULL == "full"
    assert PATCH_VECTOR_MODE_SCAFFOLDING != PATCH_VECTOR_MODE_FULL


def test_default_is_scaffolding():
    assert DEFAULT_PATCH_VECTOR_MODE == PATCH_VECTOR_MODE_SCAFFOLDING
