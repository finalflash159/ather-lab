"""Canonical identities and explicit, checksum-protected scenario persistence."""

import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ather_exploration.types import Scenario


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def scenario_hash(scenario):
    data = asdict(scenario)
    data.pop("witness_reference")
    return digest(data)


def geometry_hash(terrain):
    array = np.array([list(row) for row in terrain])
    variants = []
    for i in range(4):
        rotated = np.rot90(array, i)
        for candidate in (rotated, np.fliplr(rotated)):
            variants.append(tuple("".join(row) for row in candidate))
    return digest(min(variants))


def world_hash(scenario):
    data = asdict(scenario)
    for key in ("spawn", "seed", "witness_reference"):
        data.pop(key)
    return digest(data)


def implementation_id():
    # Include every subpackage and key by relative path, not basename: app.py
    # in two packages must not collide. Installation location is not identity.
    root = Path(__file__).resolve().parents[1]
    content = {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*.py"))
    }
    # Include the upstream grid implementation actually used by the runtime.
    from minigrid.core import grid, world_object

    for module in (grid, world_object):
        content[module.__name__] = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    content["numpy"] = np.__version__
    return digest(content)


def scenario_from_dict(data):
    data = dict(data)
    for key in ("terrain", "spawn", "phases"):
        data[key] = tuple(data[key])
    for key in ("pois", "room_labels", "topology_edges"):
        if key in data:
            data[key] = tuple(tuple(row) for row in data[key])
    data["routes"] = tuple(tuple(tuple(pos) for pos in route) for route in data["routes"])
    return Scenario(**data)


def read_record(path):
    record = json.loads(Path(path).read_text())
    if not isinstance(record, dict):
        raise TypeError("Record must be a JSON object")
    if set(record) != {"state", "checksum", "payload"} or record["state"] != "READY":
        raise ValueError("Record is incomplete or has an unsupported schema")
    if digest(record["payload"]) != record["checksum"]:
        raise ValueError("Record checksum mismatch")
    return record["payload"]


def write_record(path, payload, *, replace=False):
    """Publish a complete file atomically; immutable by default (including races)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"state": "READY", "checksum": digest(payload), "payload": payload}
    encoded = json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            try:
                os.link(temporary, path)
            except FileExistsError:
                if read_record(path) != json.loads(json.dumps(payload)):
                    raise ValueError(
                        f"Refusing to overwrite different immutable record: {path}"
                    ) from None
    finally:
        Path(temporary).unlink(missing_ok=True)
