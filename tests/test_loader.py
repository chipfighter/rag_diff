"""Tests for the adapter loader module."""

import pytest

from rag_diff.utils.loader import load_adapter, parse_target


class TestParseTarget:
    def test_simple_module_and_function(self):
        module, func = parse_target("my_module:my_func")
        assert module == "my_module"
        assert func == "my_func"

    def test_dotted_module_path(self):
        module, func = parse_target("my.package.module:my_func")
        assert module == "my.package.module"
        assert func == "my_func"

    def test_file_path_target(self):
        module, func = parse_target("path/to/adapter.py:my_func")
        assert module == "path/to/adapter.py"
        assert func == "my_func"

    def test_no_colon_raises_value_error(self):
        with pytest.raises(ValueError, match="module:function"):
            parse_target("no_colon_here")

    def test_empty_module_raises(self):
        with pytest.raises(ValueError, match="Module"):
            parse_target(":func")

    def test_empty_function_raises(self):
        with pytest.raises(ValueError, match="Function"):
            parse_target("module:")


class TestLoadAdapter:
    def test_load_from_file_path(self, tmp_path):
        adapter_file = tmp_path / "my_adapter.py"
        adapter_file.write_text(
            "async def ask(query: str) -> dict:\n"
            "    return {'answer': 'test', 'contexts': []}\n"
        )
        func = load_adapter(f"{adapter_file}:ask")
        assert callable(func)

    def test_load_from_python_module(self):
        func = load_adapter("os.path:join")
        assert callable(func)

    def test_missing_module_raises_import_error(self):
        with pytest.raises(ImportError):
            load_adapter("nonexistent_module_xyz_999:func")

    def test_missing_function_raises_attribute_error(self):
        with pytest.raises(AttributeError):
            load_adapter("os:nonexistent_function_xyz_999")

    def test_non_callable_raises_type_error(self):
        with pytest.raises(TypeError, match="not callable"):
            load_adapter("os:name")

    def test_file_not_found_raises(self):
        with pytest.raises(ImportError, match="not found"):
            load_adapter("/nonexistent/path/adapter.py:ask")

    def test_load_sync_function_from_file(self, tmp_path):
        adapter_file = tmp_path / "sync_adapter.py"
        adapter_file.write_text(
            "def ask(query: str) -> dict:\n"
            "    return {'answer': query, 'contexts': ['ctx']}\n"
        )
        func = load_adapter(f"{adapter_file}:ask")
        assert callable(func)
        result = func("hello")
        assert result["answer"] == "hello"
