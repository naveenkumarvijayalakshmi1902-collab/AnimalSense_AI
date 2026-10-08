from __future__ import annotations

import shutil
from pathlib import Path


def locate_tasks() -> Path:
    import ultralytics
    p = Path(ultralytics.__file__).resolve().parent / "nn" / "tasks.py"
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def main() -> None:
    path = locate_tasks()
    text = path.read_text(encoding="utf-8")

    # Import CBAM AFTER the future import.
    if "from ultralytics.nn.modules.conv import CBAM" not in text:
        marker = "from __future__ import annotations\n"
        if marker not in text:
            raise RuntimeError("Could not locate future import.")
        text = text.replace(
            marker,
            marker + "\nfrom ultralytics.nn.modules.conv import CBAM\n",
            1,
        )

    if "if m is CBAM:" not in text:
        target = "        if m in base_modules:\n"
        special = (
            "        if m is CBAM:\n"
            "            # CBAM preserves the input channel count.\n"
            "            c1 = ch[f]\n"
            "            c2 = c1\n"
            "            args = [c1, *args]\n"
            "        elif m in base_modules:\n"
        )

        if target not in text:
            raise RuntimeError(
                "Could not locate Ultralytics parse_model() base_modules branch. "
                "Check the installed version."
            )

        backup = path.with_suffix(path.suffix + ".animalsense.bak2")
        if not backup.exists():
            shutil.copy2(path, backup)

        text = text.replace(target, special, 1)

    path.write_text(text, encoding="utf-8")
    print(f"CBAM patch applied successfully to {path}")


if __name__ == "__main__":
    main()
