"""Shared XML and scalar validation for explicitly supplied local paths."""
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


def read_xml(path):
    data = Path(path).read_bytes()
    declarations = data.replace(b"\x00", b"").upper()
    if b"<!DOCTYPE" in declarations or b"<!ENTITY" in declarations:
        raise ValueError("DTD/entity declarations are not supported")
    try:
        root = ET.fromstring(data)
    except ET.ParseError as error:
        raise ValueError(f"invalid XML: {error}") from error
    for element in root.iter():
        element.tag = element.tag.rsplit("}", 1)[-1]
    return root


def positive_number(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be positive and finite")
    try:
        number = float(value)
    except (ValueError, TypeError) as error:
        raise ValueError(f"{name} must be positive and finite") from error
    if not np.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return number


def integer(value, name, minimum=0):
    # No silent truncation of fractional frames or acceptance of booleans.
    if isinstance(value, (bool, np.bool_)) or not str(value).strip().isascii():
        raise ValueError(f"{name} must be an integer >= {minimum}")
    text = str(value).strip()
    if not text.isdecimal():
        raise ValueError(f"{name} must be an integer >= {minimum}")
    number = int(text)
    if number < minimum or number > np.iinfo(np.int64).max:
        raise ValueError(f"{name} must be an integer >= {minimum} within int64 range")
    return number


def required_text(parent, path):
    value = parent.findtext(path)
    if value is None or not value.strip():
        raise ValueError(f"missing {path}")
    return value.strip()
