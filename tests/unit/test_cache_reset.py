"""Cache reset must not remove models, the index or saved conversation artifacts."""

from app.config.settings import Settings
from scripts.clear_caches import clear_response_caches


def test_clear_caches_only_removes_managed_response_files(tmp_path):
    settings = Settings(_env_file=None, data_dir=tmp_path)
    for name in ("evidence", "wording", "reviews", "tokenizers"):
        directory = tmp_path / "cache" / name
        directory.mkdir(parents=True)
        (directory / "one.json").write_text("{}")
    (tmp_path / "cache/evidence/write.tmp").write_text("pending")
    model = tmp_path / "models" / "weights.bin"
    model.parent.mkdir()
    model.write_bytes(b"model")
    assert clear_response_caches(settings, dry_run=True)["files"] == {
        "evidence": 2,
        "wording": 1,
        "reviews": 1,
    }
    assert (tmp_path / "cache/evidence/one.json").exists()
    assert clear_response_caches(settings)["files"] == {"evidence": 2, "wording": 1, "reviews": 1}
    assert not (tmp_path / "cache/evidence/one.json").exists()
    assert (tmp_path / "cache/tokenizers/one.json").exists()
    assert model.read_bytes() == b"model"
