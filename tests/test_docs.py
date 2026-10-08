"""The documentation must describe the repository that exists.

Two kinds of claim rot silently when directories move, and both bit this
repository during the reorganisation into ``analysis/``:

* a relative path that no longer resolves - and a path written between backticks
  rather than as a Markdown link is invisible to a link checker;
* a documented command-line option that the script does not have.

These checks are cheap and they run in CI, so neither can come back unnoticed.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from sdhcal import paths

REPO = paths.repo_root()

#: `report/` is a LaTeX project with its own conventions and its own asset
#: directories; it is self-contained and not checked here.
SKIPPED_DIRS = {".git", "report", ".venv", "output", "data"}

#: Tokens that are patterns or placeholders rather than paths.
PLACEHOLDERS = ("*", "{", "<", "$")


def markdown_files() -> list[Path]:
    return sorted(
        path
        for path in REPO.rglob("*.md")
        if not SKIPPED_DIRS & set(path.relative_to(REPO).parts)
    )


def test_there_are_documentation_files_to_check():
    """Guard against the discovery itself silently finding nothing."""
    found = markdown_files()
    assert len(found) > 10, f"only {len(found)} markdown files found"


@pytest.mark.parametrize("markdown", markdown_files(), ids=lambda p: str(p.relative_to(REPO)))
def test_markdown_links_resolve(markdown: Path):
    """Every relative Markdown link points at something that exists."""
    broken = [
        target
        for target in re.findall(r"\]\((?!https?:|mailto:|#)([^)#]+)", markdown.read_text(encoding="utf-8"))
        if not (markdown.parent / target).exists()
    ]
    assert not broken, f"{markdown.relative_to(REPO)}: broken link(s) {broken}"


@pytest.mark.parametrize("markdown", markdown_files(), ids=lambda p: str(p.relative_to(REPO)))
def test_backticked_relative_paths_resolve(markdown: Path):
    """A relative path in backticks must resolve too.

    Only paths that climb out of their own directory (``../``) are checked: those
    are the ones whose depth changes when a file moves, and the ones a Markdown
    link checker never sees.
    """
    broken = []
    for token in re.findall(r"`((?:\.\./)+[^`\s]+)`", markdown.read_text(encoding="utf-8")):
        if any(placeholder in token for placeholder in PLACEHOLDERS):
            continue
        if not (markdown.parent / token.rstrip("/")).exists():
            broken.append(token)
    assert not broken, f"{markdown.relative_to(REPO)}: broken backticked path(s) {broken}"


def _declared_options(script: Path) -> set[str]:
    """Long options passed to add_argument in a script."""
    options = set()
    for node in ast.walk(ast.parse(script.read_text(encoding="utf-8"))):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
        ):
            for argument in node.args:
                if isinstance(argument, ast.Constant) and str(argument.value).startswith("--"):
                    options.add(argument.value)
    return options


def _documented_commands() -> list[tuple[Path, str]]:
    commands = []
    for markdown in markdown_files():
        for line in markdown.read_text(encoding="utf-8").splitlines():
            stripped = line.strip().lstrip("$ ")
            if stripped.startswith("python ") and ".py" in stripped:
                commands.append((markdown, stripped))
    return commands


def test_documented_commands_name_scripts_that_exist():
    missing = []
    for markdown, command in _documented_commands():
        script = next((part for part in command.split() if part.endswith(".py")), None)
        if script is None:
            continue
        # A command shown in a README is run from that README's directory.
        if not ((markdown.parent / script).is_file() or (REPO / script).is_file()):
            missing.append(f"{markdown.relative_to(REPO)}: {script}")
    assert not missing, "documented scripts that do not exist: " + ", ".join(missing)


def test_documented_options_exist():
    """A README that advertises --foo must name a script that accepts --foo."""
    problems = []
    for markdown, command in _documented_commands():
        parts = command.split()
        script = next((part for part in parts if part.endswith(".py")), None)
        if script is None:
            continue
        path = next(
            (candidate for candidate in ((markdown.parent / script), (REPO / script)) if candidate.is_file()),
            None,
        )
        if path is None:
            continue
        options = _declared_options(path)
        for token in parts:
            if token.startswith("--") and token not in options:
                problems.append(f"{markdown.relative_to(REPO)}: {path.name} has no {token}")
    assert not problems, "undocumented options: " + "; ".join(sorted(set(problems)))


def test_makefile_references_existing_files():
    """Every repository path named in the Makefile exists."""
    makefile = (REPO / "Makefile").read_text(encoding="utf-8")
    referenced = set()
    for match in re.finditer(r"\b((?:analysis|cpp|src|tools|tests|calibration)/[\w./-]+)", makefile):
        # Trailing sentence punctuation is not part of the path.
        token = match.group(1).rstrip(".,;:")
        if any(placeholder in token for placeholder in PLACEHOLDERS):
            continue
        referenced.add(token)

    # Build products are created by the targets, not committed.
    generated = {"analysis/1_shower_features/computeParams", "analysis/1_shower_features/computeParams_parallel"}
    missing = sorted(token for token in referenced - generated if not (REPO / token).exists())
    assert not missing, f"Makefile refers to missing path(s): {missing}"
    assert len(referenced) > 5, "the Makefile path scan found suspiciously little"


def test_no_absolute_internship_paths_outside_the_translation_layer():
    """The old machine's paths may appear only where they are being translated."""
    allowed = {
        Path("src/sdhcal/paths.py"),  # the prefix table
        Path("tests/test_paths.py"),  # its tests
        Path("tests/test_docs.py"),  # this file
    }
    offenders = []
    for source in REPO.rglob("*.py"):
        relative = source.relative_to(REPO)
        if SKIPPED_DIRS & set(relative.parts) or relative in allowed:
            continue
        for number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
            if "/gridgroup/" not in line:
                continue
            stripped = line.strip()
            # Docstrings and comments may quote the old path when explaining a fix,
            # and legacy() keeps the literal on purpose.
            if stripped.startswith(("#", "*", '"', "'")) or "legacy(" in line:
                continue
            offenders.append(f"{relative}:{number}")
    assert not offenders, "unwrapped absolute paths: " + ", ".join(offenders)


def _requirement_names(lines) -> set[str]:
    """Distribution names from a requirements-style listing, ignoring comments."""
    names = set()
    for line in lines:
        line = line.split("#")[0].strip()
        if not line or line.startswith("-"):
            continue
        names.add(re.split(r"[<>=!\[;]", line)[0].strip().lower())
    return names


def test_requirements_and_pyproject_agree():
    """Two dependency lists are two chances to drift; they must declare the same set.

    ``requirements.txt`` is the file a reader runs; ``pyproject.toml`` is what
    ``pip install -e .`` uses. If they disagree, one environment gets a package
    the other does not - which is how seaborn ended up imported by eight scripts
    while being absent from the installable metadata.
    """
    import tomllib

    pyproject = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    declared = {
        re.split(r"[<>=!\[]", dep)[0].strip().lower()
        for dep in pyproject["project"]["dependencies"]
    }
    extras = {
        name.lower()
        for group in pyproject["project"].get("optional-dependencies", {}).values()
        for name in (re.split(r"[<>=!\[]", dep)[0].strip() for dep in group)
    }

    listed = _requirement_names((REPO / "requirements.txt").read_text(encoding="utf-8").splitlines())
    # requirements.txt also carries the development tools, which live in an extra.
    runtime = listed - extras

    assert runtime == declared, (
        f"requirements.txt only: {sorted(runtime - declared)}; "
        f"pyproject only: {sorted(declared - runtime)}"
    )


def test_every_third_party_import_is_declared():
    """A package imported by any script must be installable from the metadata."""
    import tomllib

    pyproject = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    declared = {
        re.split(r"[<>=!\[]", dep)[0].strip().lower()
        for dep in pyproject["project"]["dependencies"]
    }
    for group in pyproject["project"].get("optional-dependencies", {}).values():
        declared |= {re.split(r"[<>=!\[]", dep)[0].strip().lower() for dep in group}

    # Import name -> distribution name, plus what is not pip-installable.
    aliases = {"sklearn": "scikit-learn", "imblearn": "imbalanced-learn", "torch_geometric": "torch-geometric"}
    not_on_pypi = {"root"}  # ROOT's own Python bindings
    import sys

    standard_library = set(sys.stdlib_module_names)

    undeclared: dict[str, str] = {}
    for source in REPO.rglob("*.py"):
        relative = source.relative_to(REPO)
        if SKIPPED_DIRS & set(relative.parts):
            continue
        for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                modules = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules = [node.module.split(".")[0]]
            else:
                continue
            for module in modules:
                if module in standard_library or module == "sdhcal" or module.startswith("_"):
                    continue
                name = aliases.get(module, module).lower()
                if name in declared or name in not_on_pypi:
                    continue
                undeclared.setdefault(name, str(relative))

    assert not undeclared, "imported but not declared: " + ", ".join(
        f"{name} (e.g. {where})" for name, where in sorted(undeclared.items())
    )
