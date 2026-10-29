"""main.py must not re-import modules inside main(): a local `import pandas as pd`
in one --mode branch makes `pd` local to the whole function, and every other
branch that uses `pd` (paper-run, reconcile) then crashes with UnboundLocalError."""

import ast
from pathlib import Path


def test_main_has_no_function_level_import_of_module_level_names():
    tree = ast.parse(Path("main.py").read_text())
    top = {a.asname or a.name for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))
           for a in n.names}
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    shadowed = {a.asname or a.name for n in ast.walk(main)
                if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names} & top
    assert not shadowed, f"re-imported inside main(): {shadowed}"
