from __future__ import annotations

import builtins
import importlib.util
import sys
from pathlib import Path


def _load_apigen_module():
    repo_root = Path(__file__).resolve().parents[2]
    apigen_path = repo_root / "docs" / "source" / "ext" / "apigen.py"
    spec = importlib.util.spec_from_file_location("fury_docs_apigen", apigen_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_apigen_passes_utf8_encoding_for_text_open(monkeypatch, tmp_path):
    apigen = _load_apigen_module()

    # Create a module containing a UTF-8 sequence that includes byte 0x8F.
    # U+03CF encodes to UTF-8 as: 0xCF 0x8F.
    module_name = "_fury_apigen_encoding_smoke"
    module_file = tmp_path / f"{module_name}.py"
    module_file.write_text(
        '# -*- coding: utf-8 -*-\n'
        '"""Module docstring with a non-ASCII char: \u03cf."""\n\n'
        "def foo():\n"
        "    return 1\n",
        encoding="utf-8",
    )

    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop(module_name, None)
    __import__(module_name)

    real_open = builtins.open
    open_calls: list[tuple[tuple, dict]] = []

    def strict_text_open(file, mode="r", *args, **kwargs):
        # Enforce explicit encoding for text-mode open.
        if "b" not in mode and kwargs.get("encoding") is None:
            raise AssertionError("expected encoding= for text open()")
        open_calls.append(((file, mode, *args), dict(kwargs)))
        return real_open(file, mode, *args, **kwargs)

    # Patch AFTER importing the module so we don't interfere with import machinery.
    monkeypatch.setattr(builtins, "open", strict_text_open)

    dw = apigen.ApiDocWriter.__new__(apigen.ApiDocWriter)
    dw.object_skip_patterns = []
    functions, classes, constants = dw._parse_module_with_import(module_name)

    assert isinstance(functions, list)
    assert isinstance(classes, list)
    assert isinstance(constants, list)

    # Also exercise a write path that previously defaulted to locale encoding.
    dw.written_modules = ["dummy.rst"]
    dw.rst_extension = ".rst"
    dw.write_index(str(tmp_path), froot="gen", relative_to=None)

    assert any(
        (str(module_file) in str(args[0]) and kwargs.get("encoding") == "utf-8")
        for (args, kwargs) in open_calls
    )

    assert any(
        (str(tmp_path) in str(args[0]) and kwargs.get("encoding") == "utf-8")
        for (args, kwargs) in open_calls
    )
