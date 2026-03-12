# ml-antiviral-diagnosis
### by Stavros Tseranidis
### A case study for an alert system for antiviral treatment



## Local development

This project uses `uv` with `pyproject.toml` for dependency management.

### 1) Install uv globally

- Windows (Git Bash): `py -m pip install uv`
- macOS: `python3 -m pip install uv`

### 2) Create and activate virtual environment


- Windows (Git Bash):
	- `uv venv .venv`
	- `source .venv/Scripts/activate`
- macOS (zsh/bash):
	- `uv venv .venv`
	- `source .venv/bin/activate`

### 3) Install dependencies

- Main deps only: `uv sync`
- Include dev deps (notebooks + linting): `uv sync --extra dev`

### 4) Use the environment in notebooks (VS Code)

1. First set the Python interpreter to `.venv`:
	- Open Command Palette (`Ctrl+Shift+P`, or `Ctrl+P` then type `>`).
	- Run `Python: Select Interpreter`.
	- Choose the `.venv` interpreter from this project.
2. Open a notebook in `notebooks/`.
3. Select kernel from top-right and choose the `.venv` / project interpreter.

### Linting and notebooks

- For `.py` files (with auto formatting):
    - `uv run ruff check . --fix`
    - `uv run black .`
- For `.ipynb`, run through `nbqa`:
	- `uv run nbqa ruff notebooks/`
	- `uv run nbqa black notebooks/`
