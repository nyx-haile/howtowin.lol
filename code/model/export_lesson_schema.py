"""Export site-facing lesson schema lock.

Emits site/src/lib/lesson/schema.lock.json with a stable-sorted description of
AnchorCandidate + LessonResult so the TS types can be checked for drift.
Run: `python -m model.export_lesson_schema`
"""
import dataclasses
import json
import os
import typing
from typing import get_args, get_origin

from model.lesson import AnchorCandidate, LessonResult


def _describe_type(tp) -> dict:
    origin = get_origin(tp)
    args = get_args(tp)
    if origin is None:
        name = getattr(tp, "__name__", str(tp))
        return {"kind": "primitive", "name": name}
    if origin is typing.Union:
        non_none = [a for a in args if a is not type(None)]
        optional = len(non_none) != len(args)
        if len(non_none) == 1:
            inner = _describe_type(non_none[0])
            return {"kind": "optional", "of": inner} if optional else inner
        return {"kind": "union", "of": [_describe_type(a) for a in non_none], "optional": optional}
    if origin in (list, typing.List):
        return {"kind": "list", "of": _describe_type(args[0])}
    return {"kind": "primitive", "name": str(tp)}


def _describe(cls) -> dict:
    return {
        "name": cls.__name__,
        "fields": sorted(
            (
                {"name": f.name, "type": _describe_type(f.type)}
                for f in dataclasses.fields(cls)
            ),
            key=lambda f: f["name"],
        ),
    }


def main() -> None:
    lock = {
        "schemaVersion": 1,
        "source": "code/model/lesson.py",
        "dataclasses": [_describe(AnchorCandidate), _describe(LessonResult)],
    }
    out_path = os.path.join(
        os.path.dirname(__file__), "..", "..",
        "site", "src", "lib", "lesson", "schema.lock.json",
    )
    out_path = os.path.abspath(out_path)
    with open(out_path, "w") as f:
        json.dump(lock, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
