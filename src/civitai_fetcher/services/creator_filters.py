"""
Simple, human-editable creator block list.

File format (one entry per line):
    FutaWorld: block

Comparison against posterUsername (the image uploader) is case-insensitive.
Blank lines and lines starting with # are ignored. Only "block" is a
meaningful action right now -- anything else is logged and ignored
(reserved in case an "allow"-only allowlist mode is wanted later).
"""
import os


def load_creator_filters(path):
    """
    Returns {username_lower: action_lower}. A missing file is not an error
    -- just an empty dict, meaning "block nothing".
    """
    filters = {}
    if not os.path.exists(path):
        return filters

    with open(path, encoding="utf-8") as f:
        for lineno, raw_line in enumerate(f, 1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                print(f"  [creator_filters] {path}:{lineno}: skipping malformed line "
                      f"(expected 'username: action'): {raw_line.strip()}")
                continue
            username, action = line.split(":", 1)
            username = username.strip().lower()
            action = action.strip().lower()
            if not username:
                continue
            if action != "block":
                print(f"  [creator_filters] {path}:{lineno}: unknown action '{action}' for "
                      f"'{username}', ignoring (only 'block' is supported)")
                continue
            filters[username] = action

    return filters


def is_blocked(username, filters):
    if not username:
        return False
    return filters.get(username.strip().lower()) == "block"