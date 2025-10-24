import ast
import tempfile
import subprocess
import os
import sys
from difflib import SequenceMatcher


def is_syntax_valid_python(code: str) -> bool:
    """Check Python syntax by parsing with ast."""
    try:
        ast.parse(code)
        return True
    except Exception:
        return False


def run_py_compile_file(path: str) -> bool:
    """Try to compile a Python file with py_compile (checks syntax)."""
    try:
        import py_compile

        py_compile.compile(path, doraise=True)
        return True
    except Exception:
        return False


def run_code_in_subprocess(code: str, timeout: int = 5) -> (bool, str):
    """Run code in subprocess safely (best-effort). Returns (success, stdout+stderr).

    Note: Running arbitrary code can be unsafe. This helper uses subprocess with a timeout
    and should be run in a sandbox if you don't trust the inputs.
    """
    with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False, encoding='utf-8') as f:
        f.write(code)
        tmp = f.name

    try:
        proc = subprocess.run([sys.executable, tmp], capture_output=True, text=True, timeout=timeout)
        ok = proc.returncode == 0
        out = (proc.stdout or '') + (proc.stderr or '')
        return ok, out
    except subprocess.TimeoutExpired as e:
        return False, f"Timeout after {timeout}s"
    except Exception as e:
        return False, str(e)
    finally:
        try:
            os.remove(tmp)
        except Exception:
            pass


def compute_simple_code_similarity(pred: str, ref: str) -> float:
    """Fallback similarity: sequence matcher ratio on whitespace-normalized code.

    This is a lightweight substitute for CodeBLEU / full semantic metrics and works
    well enough for quick feedback in the smoke tests.
    """
    a = "\n".join([ln.strip() for ln in pred.splitlines() if ln.strip()])
    b = "\n".join([ln.strip() for ln in ref.splitlines() if ln.strip()])
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def evaluate_generated_code(pred: str, ref: str, run_tests: bool = False) -> dict:
    """Compute a small set of signals for a generated Python snippet.

    Returns dict with keys: syntax_ok (bool), similarity (0..1), exec_ok (bool), exec_output (str)
    """
    syntax_ok = is_syntax_valid_python(pred)
    similarity = compute_simple_code_similarity(pred, ref)
    exec_ok = False
    exec_out = ""
    if run_tests and syntax_ok:
        exec_ok, exec_out = run_code_in_subprocess(pred)

    return {
        'syntax_ok': syntax_ok,
        'similarity': similarity,
        'exec_ok': exec_ok,
        'exec_output': exec_out,
    }
