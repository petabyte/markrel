# Release Guide 🐟

## Creating a New Release

### 1. Update Version

Update `pyproject.toml`:
```toml
[project]
name = "markrel"
version = "0.2.0"  # Bump version
```

### 2. Update Changelog

Add to README.md or create CHANGELOG.md:
```markdown
## v0.2.0 (2026-03-23)
- New feature: X
- Bug fix: Y
- Performance improvement: Z
```

### 3. Commit Changes

```bash
git add pyproject.toml README.md
git commit -m "Bump version to 0.2.0"
git push origin master
```

### 4. Create Tag

```bash
git tag -a v0.2.0 -m "Release version 0.2.0"
git push origin v0.2.0
```

### 5. CI/CD Takes Over

The GitHub Actions workflow will automatically:
- ✅ Build the package
- ✅ Upload to PyPI
- ✅ Create GitHub Release with notes

### 6. Verify

- Check PyPI: https://pypi.org/project/markrel/
- Check GitHub Releases: https://github.com/petabyte/markrel/releases

## Manual Release (if needed)

```bash
# Build
python -m build

# Test
 twine check dist/*

# Upload to PyPI
twine upload dist/*
```

## Version Numbering

Follows [Semantic Versioning](https://semver.org/):
- MAJOR: Breaking changes (v1.0.0, v2.0.0)
- MINOR: New features, backwards compatible (v0.1.0, v0.2.0)
- PATCH: Bug fixes (v0.1.1, v0.1.2)

## PyPI Setup

Before first release, create a PyPI API token:

1. Go to https://pypi.org/manage/account/token/
2. Generate token with "Upload to PyPI" scope
3. Add to GitHub Secrets:
   - Go to Settings → Secrets → Actions
   - Create `PYPI_API_TOKEN`

---

Happy releasing! 🐟
