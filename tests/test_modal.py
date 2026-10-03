import hashlib
import json
from types import SimpleNamespace

import pytest

from ather_exploration.training.checkpoints import FILES
from ather_exploration.training.modal_io import download_run, fetch, identifier
from ather_exploration.worlds.scenarios import implementation_id, read_record, write_record


class FakeVolume:
    def __init__(self, root):
        self.root = root

    def read_file_into_fileobj(self, path, stream):
        return stream.write((self.root / path.lstrip("/")).read_bytes())

    def iterdir(self, path):
        return [
            SimpleNamespace(path=str(p.relative_to(self.root)))
            for p in (self.root / path.lstrip("/")).rglob("*")
            if p.is_file()
        ]


def remote_run(root):
    run = root / "runs" / "test"
    checkpoint = run / "checkpoints" / "step_256"
    checkpoint.mkdir(parents=True)
    for name in FILES:
        (checkpoint / name).write_bytes(b"test transport bytes, not a serialized model")
    (checkpoint / "metadata.json").write_text(
        json.dumps(
            {
                "artifact_schema": "g4-checkpoint-v1",
                "source_revision": implementation_id(),
                "method": "A",
            }
        )
    )
    (checkpoint / "checksums.json").write_text(
        json.dumps(
            {name: hashlib.sha256((checkpoint / name).read_bytes()).hexdigest() for name in FILES}
        )
    )
    (checkpoint / "READY").write_text("ready")
    write_record(run / "latest.json", {"checkpoint": "checkpoints/step_256"})
    (run / "progress.jsonl").write_bytes(b'{"step":256}\n{"partial":')
    return run, checkpoint


def test_sync_publishes_verified_checkpoint_and_complete_log_rows(tmp_path):
    remote, _ = remote_run(tmp_path / "remote")
    volume = FakeVolume(tmp_path / "remote")
    output = tmp_path / "local"
    download_run(volume, "test", output)
    assert read_record(output / "latest.json")["checkpoint"] == "checkpoints/step_256"
    assert (output / "progress.jsonl").read_bytes() == b'{"step":256}\n'
    download_run(volume, "test", output)  # Immutable checkpoint reused, no deserialization.
    assert (remote / "latest.json").exists()


def test_corrupt_checkpoint_never_publishes_latest(tmp_path):
    _, checkpoint = remote_run(tmp_path / "remote")
    (checkpoint / "model.zip").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        download_run(FakeVolume(tmp_path / "remote"), "test", tmp_path / "local")
    assert not (tmp_path / "local/latest.json").exists()


def test_remote_pointer_cannot_escape(tmp_path):
    run, _ = remote_run(tmp_path / "remote")
    write_record(run / "latest.json", {"checkpoint": "../../escape"}, replace=True)
    with pytest.raises(ValueError, match="pointer"):
        download_run(FakeVolume(tmp_path / "remote"), "test", tmp_path / "local")


def test_failed_download_preserves_existing_file(tmp_path):
    class Broken:
        def read_file_into_fileobj(self, path, stream):
            stream.write(b"partial")
            raise ConnectionError("network")

    target = tmp_path / "file"
    target.write_bytes(b"old")
    with pytest.raises(ConnectionError):
        fetch(Broken(), "/remote", target)
    assert target.read_bytes() == b"old"
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize("value", ["../x", "/x", "a/b", "", "x" * 97])
def test_identifier_rejects_paths(value):
    with pytest.raises(ValueError):
        identifier(value)


def test_sync_cannot_overwrite_local_run(tmp_path):
    (tmp_path / "manifest.json").write_text("existing run")
    with pytest.raises(ValueError, match="empty"):
        download_run(FakeVolume(tmp_path), "test", tmp_path)


def test_publish_to_volume_without_hardlinks(tmp_path, monkeypatch):
    from ather_exploration.training.modal_io import publish_run

    source, _ = remote_run(tmp_path / "source")

    def unsupported(*args):
        raise PermissionError("Volume has no hardlinks")

    monkeypatch.setattr("os.link", unsupported)
    target = tmp_path / "volume/runs/test"
    publish_run(source, target)
    assert read_record(target / "latest.json")["checkpoint"] == "checkpoints/step_256"
    assert (target / "checkpoints/step_256/READY").exists()
    publish_run(source, target)
    assert not list(target.rglob(".pending-*"))


def test_download_includes_all_checkpoints_sorted_numerically(tmp_path):
    import shutil

    run, first = remote_run(tmp_path / "remote")
    second = first.parent / "step_512"
    shutil.copytree(first, second)
    write_record(run / "latest.json", {"checkpoint": "checkpoints/step_512"}, replace=True)
    output = tmp_path / "local"
    result = download_run(FakeVolume(tmp_path / "remote"), "test", output)
    assert result["checkpoints"] == 2
    assert (output / "checkpoints/step_256/READY").exists()
    assert (output / "checkpoints/step_512/READY").exists()
    assert read_record(output / "latest.json")["checkpoint"] == "checkpoints/step_512"


def test_download_recovers_after_corrupt_older_checkpoint(tmp_path):
    import shutil

    run, first = remote_run(tmp_path / "remote")
    second = first.parent / "step_512"
    shutil.copytree(first, second)
    original = (first / "model.zip").read_bytes()
    (first / "model.zip").write_bytes(b"bad")
    write_record(run / "latest.json", {"checkpoint": "checkpoints/step_512"}, replace=True)
    output = tmp_path / "local"
    with pytest.raises(ValueError, match="checksum"):
        download_run(FakeVolume(tmp_path / "remote"), "test", output)
    assert not (output / "latest.json").exists()
    (first / "model.zip").write_bytes(original)
    assert download_run(FakeVolume(tmp_path / "remote"), "test", output)["checkpoints"] == 2


def test_fetch_uses_offset_writer_not_stream_iterator(tmp_path):
    class Volume:
        def read_file(self, path):
            raise AssertionError("Corrupt streaming API must not be used")

        def read_file_into_fileobj(self, path, stream):
            stream.seek(3)
            stream.write(b"def")
            stream.seek(0)
            stream.write(b"abc")
            return 6

    target = tmp_path / "model.zip"
    fetch(Volume(), "/checkpoint/model.zip", target)
    assert target.read_bytes() == b"abcdef"


def test_transport_only_revision_allows_inference_not_resume(tmp_path):
    from ather_exploration.training.checkpoints import inspect_checkpoint

    _, cp = remote_run(tmp_path)
    metadata = json.loads((cp / "metadata.json").read_text())
    metadata["source_revision"] = "17e3e864172cc7cd52e2a70cc6b293258192b0f39b9d748aa643e4de790797c8"
    (cp / "metadata.json").write_text(json.dumps(metadata))
    hashes = json.loads((cp / "checksums.json").read_text())
    hashes["metadata.json"] = hashlib.sha256((cp / "metadata.json").read_bytes()).hexdigest()
    (cp / "checksums.json").write_text(json.dumps(hashes))
    inspect_checkpoint(cp, inference=True)
    with pytest.raises(ValueError, match="source revision"):
        inspect_checkpoint(cp)
