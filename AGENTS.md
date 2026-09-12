# Repository agent instructions

Follow [CLAUDE.md](CLAUDE.md) for repository architecture, conventions, and workflows.

Dependency source of truth is `pyproject.toml`; do not add or rely on a checked-in `requirements.txt`. For editable development installs, select one runtime extra explicitly, for example `uv pip install -e ".[onnx-cpu]"` (recommended), `uv pip install -e ".[onnx-gpu]"`, or `uv pip install -e ".[paddle-gpu]"`. The Vercel deployment script (`deploy_vercel.sh`) generates a temporary Linux CPU `requirements.txt` from the project metadata and removes it when deployment finishes.
