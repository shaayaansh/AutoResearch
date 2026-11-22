#!/usr/bin/env bash
# Create the project virtualenv (.venv) and install dependencies into it.
# acl-anthology needs Python > 3.11 and < 3.15, so we look for a suitable interpreter.
set -euo pipefail
cd "$(dirname "$0")"

ok() { "$1" -c 'import sys; sys.exit(0 if (3,11,1) <= sys.version_info[:3] < (3,15) else 1)' 2>/dev/null; }

PY=""
for cand in ${PYTHON:-} python3.13 python3.12 python3.11 python python3; do
  if command -v "$cand" >/dev/null 2>&1 && ok "$cand"; then PY=$(command -v "$cand"); break; fi
done

if [ -z "$PY" ]; then
  echo "No Python 3.11.1-3.14 found. Install one, e.g.:"
  echo "  brew install python@3.12        (then re-run ./setup.sh)"
  echo "  or: conda create -y -n py312 python=3.12 && PYTHON=\$(conda run -n py312 which python) ./setup.sh"
  exit 1
fi
echo "Using $PY ($("$PY" --version))"

if [ -d .venv ] && ! ok .venv/bin/python; then
  echo "Existing .venv uses an unsupported Python; recreating it."
  mv .venv ".venv.old.$(date +%s)"
fi
[ -d .venv ] || "$PY" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

echo
echo "Done. Activate with:  source .venv/bin/activate"
