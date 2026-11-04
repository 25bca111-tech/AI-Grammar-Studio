#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# setup-python.sh — idempotent bootstrap for the Python NLP engine.
# Wired into npm lifecycle hooks (predev / prestart / postinstall) so a fresh
# clone or deployment ALWAYS has nltk + textblob available to whichever
# python3 the API route ends up spawning. Never blocks npm: exits 0.
# ---------------------------------------------------------------------------
cd "$(dirname "$0")/.." || exit 0

for PY in python3 python; do
  command -v "$PY" >/dev/null 2>&1 || continue

  if "$PY" -c "import nltk, textblob" >/dev/null 2>&1; then
    echo "[setup-python] $PY: nltk + textblob already present ✓"
    exit 0
  fi

  echo "[setup-python] installing Python deps into $PY (one time, ~10 MB)..."
  "$PY" -m pip install -q -r pyengine/requirements.txt 2>/dev/null \
    || "$PY" -m pip install -q --user -r pyengine/requirements.txt 2>/dev/null \
    || "$PY" -m pip install -q --break-system-packages --user -r pyengine/requirements.txt 2>/dev/null \
    || true

  if "$PY" -c "import nltk, textblob" >/dev/null 2>&1; then
    echo "[setup-python] installed into $PY ✓"
    exit 0
  fi
done

echo "[setup-python] WARNING: could not auto-install Python dependencies." >&2
echo "  Fix manually:  cd pyengine && python3 -m pip install -r requirements.txt" >&2
exit 0  # informational only — never fail the npm hook
