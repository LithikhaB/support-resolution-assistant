import json
from unittest.mock import Mock

import pytest

from app.config.settings import Settings
from app.ingestion.artifacts import file_sha256
from scripts import download_dataset


def test_download_resolves_revision_before_fetching(tmp_path, monkeypatch):
    settings = Settings(_env_file=None, data_dir=tmp_path)
    monkeypatch.setattr(download_dataset, "get_settings", lambda: settings)
    monkeypatch.setattr("sys.argv", ["download_dataset", "--revision", "release"])
    api = Mock()
    api.dataset_info.return_value.sha = "a" * 40
    monkeypatch.setattr(download_dataset, "HfApi", lambda: api)
    # Use a small real object to support len() without contacting Hugging Face.
    class English:
        def __len__(self):
            return 1
        def to_csv(self, path, index):
            assert index is False
            from pathlib import Path
            Path(path).write_text("language,body\nen,An example complaint\n", encoding="utf-8")
    dataset = Mock()
    def filtered(predicate):
        assert predicate({"language": " EN "})
        assert not predicate({"language": "de"})
        assert not predicate({"language": None})
        return English()
    dataset.filter.side_effect = filtered
    fetch = Mock(return_value=dataset)
    monkeypatch.setattr(download_dataset, "load_dataset", fetch)
    download_dataset.main()
    fetch.assert_called_once_with(download_dataset.SOURCE, revision="a" * 40, split="train")
    path = settings.raw_dir / "customer_support_tickets.csv"
    manifest = json.loads(path.with_suffix(".manifest.json").read_text())
    assert manifest["revision"] == "a" * 40
    assert manifest["sha256"] == file_sha256(path)


def test_missing_revision_aborts_before_download(monkeypatch):
    monkeypatch.setattr("sys.argv", ["download_dataset"])
    api = Mock()
    api.dataset_info.return_value.sha = None
    monkeypatch.setattr(download_dataset, "HfApi", lambda: api)
    fetch = Mock()
    monkeypatch.setattr(download_dataset, "load_dataset", fetch)
    with pytest.raises(RuntimeError, match="immutable"):
        download_dataset.main()
    fetch.assert_not_called()
