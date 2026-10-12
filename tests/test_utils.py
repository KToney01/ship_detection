import json

from ship_detection.utils import save_json

def test_save_json_creates_file(tmp_path):
    output_path = tmp_path / "test.json"

    data = {
        "value": 42
    }

    result = save_json(
        data,
        output_path,
        overwrite=False
    )

    assert result is True
    assert output_path.exists()

    with output_path.open("r") as file:
        saved_data = json.load(file)

    assert saved_data == data

def test_save_json_prevents_overwrite(tmp_path):
    output_path = tmp_path / "test.json"

    original_data = {
        "value": 42
    }

    new_data = {
        "value": 99
    }

    save_json(
        original_data,
        output_path,
        overwrite=False
    )

    result = save_json(
        new_data,
        output_path,
        overwrite=False
    )

    assert result is False

    with output_path.open("r") as file:
        saved_data = json.load(file)

    assert saved_data == original_data