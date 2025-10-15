from pathlib import Path

def pytest_collection_modifyitems(config, items):
    # Explicit test file order (unit + integration)
    order_map = {
        "tests/python/unit/test_unit_downloader.py"    : 1,
        "tests/python/unit/test_unit_preprocessor.py"  : 2,
        "tests/python/unit/test_unit_dataset.py"       : 3,
        "tests/python/unit/test_unit_model.py"         : 4,
        "tests/python/unit/test_unit_trainer.py"       : 5,
        "tests/python/integration/test_dataset.py"     : 6,
        "tests/python/integration/test_trainer.py"     : 7
    }

    def sort_key(item):
        rel_path = str(Path(item.fspath).as_posix())
        rel_path = str(Path(rel_path).relative_to(config.rootpath))
        return order_map.get(rel_path, 9999)

    # Sort items according to map (default = 9999 → goes last)
    items.sort(key=sort_key)