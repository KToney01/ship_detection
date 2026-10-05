import json
from pathlib import Path

def save_json(data, path, overwrite=False):
    path = Path(path)

    if path.exists() and not overwrite:
        print(f"File already exists: {path}")
        return False
    with path.open("w") as file:
        json.dump(
            data,
            file,
            indent=4
        )
    print(f"Saved JSON to: {path}")
    return True