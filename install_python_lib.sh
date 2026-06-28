#!/usr/bin/env bash
set -euo pipefail

MODULE=fringe

cargo build --release --lib

if command -v python3 >/dev/null; then
    PYTHON=python3
elif command -v python >/dev/null; then
    PYTHON=python
else
    echo "Python not found."
    exit 1
fi

SITE_PACKAGES=$($PYTHON - <<'PY'
import site
print(site.getusersitepackages())
PY
)

case "$(uname -s)" in
    Linux*)
        cp target/release/lib${MODULE}.so \
           "${SITE_PACKAGES}/${MODULE}.so"
        ;;
    Darwin*)
        if [ -f target/release/lib${MODULE}.so ]; then
            cp target/release/lib${MODULE}.so \
               "${SITE_PACKAGES}/${MODULE}.so"
        else
            cp target/release/lib${MODULE}.dylib \
               "${SITE_PACKAGES}/${MODULE}.so"
        fi
        ;;
    MINGW*|MSYS*|CYGWIN*)
        cp target/release/${MODULE}.dll \
           "${SITE_PACKAGES}/${MODULE}.pyd"
        ;;
    *)
        echo "Unsupported platform"
        exit 1
        ;;
esac

cp ${MODULE}.pyi "${SITE_PACKAGES}/"

python3 -c "import ${MODULE}; print('Installed', ${MODULE}.__name__)"
