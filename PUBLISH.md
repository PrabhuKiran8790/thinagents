# Publishing to PyPI Checklist

## Before Publishing

### 1. Build the Frontend UI
```bash
# Build the production UI
python scripts/build_ui.py
```

This will create `thinagents/web/ui/build/` with all the production files.

### 2. Verify Build Directory Exists
```bash
# Check if build directory exists
ls -la thinagents/web/ui/build/

# Should see files like:
# - index.html
# - _app/
# - robots.txt
```

### 3. Test Local Build
```bash
# Build the package locally
python -m build

# Check what's included in the distribution
tar -tzf dist/thinagents-*.tar.gz | grep "web/ui/build"
```

You should see all the UI files listed.

### 4. Test Installation
```bash
# Create a test environment
python -m venv test_env
source test_env/bin/activate  # On Windows: test_env\Scripts\activate

# Install from local build
pip install dist/thinagents-*.whl

# Test that UI works
python -c "from pathlib import Path; import thinagents.web.backend.server as s; print(f'UI exists: {(Path(s.__file__).parent.parent / \"ui\" / \"build\").exists()}')"

# Clean up
deactivate
rm -rf test_env
```

### 5. Publish
```bash
# Upload to PyPI
python -m twine upload dist/*
```

## What Gets Included

✅ **Included:**
- `thinagents/` (all Python source code)
- `thinagents/web/ui/build/` (built UI files)
- `README.md`

❌ **Excluded:**
- `thinagents/frontend/` (source code - not needed)
- `tests/`
- `examples/`
- `scripts/`
- `node_modules/`

## Important Notes

1. **Always build UI before publishing** - Users won't have Node.js/pnpm
2. **UI build is committed to git** - The `!thinagents/web/ui/build/` in `.gitignore` ensures it's tracked
3. **Frontend source is excluded** - Users don't need the Svelte source code
4. **Dev mode still works** - If someone has the full repo with `thinagents/frontend/src/`, dev mode auto-enables

## Directory Structure in Package

```
thinagents/
├── core/
├── memory/
├── tools/
├── utils/
└── web/
    ├── backend/
    │   ├── server.py
    │   └── webui.py
    └── ui/
        └── build/          ← This gets shipped!
            ├── index.html
            ├── _app/
            └── ...
```

