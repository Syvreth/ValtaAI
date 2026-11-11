import json
from pathlib import Path

from valta.config import load_settings, save_state
from valta.status import project_status


def test_project_status_names_the_folder_and_model(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    (tmp_path / "config.toml").write_text((root / "config.toml").read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "rules").mkdir()
    (tmp_path / "rules" / "root.md").write_text((root / "rules" / "root.md").read_text(encoding="utf-8"), encoding="utf-8")
    settings = load_settings(tmp_path)
    lines = project_status(settings).splitlines()
    assert lines[0] == f"Project: {settings.root}"
    assert lines[1] == f"Personal model: {settings.self_model}"
    save_state(settings, {"personal_model": "valta", "previous_modelfile": None})
    assert project_status(settings).splitlines()[1] == "Personal model: valta"
    assert json.loads((tmp_path / "data" / "state.json").read_text(encoding="utf-8"))["personal_model"] == "valta"
