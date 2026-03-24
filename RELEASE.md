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

## PyPI Setup (Trusted Publishing)

markrel uses PyPI's **trusted publishing** (OIDC) instead of API tokens. This is more secure:

### One-Time Setup

1. Go to https://pypi.org/manage/project/markrel/settings/publishing/
2. Add a new GitHub publisher:
   - **Owner**: `petabyte`
   - **Repository**: `markrel`
   - **Workflow**: `release.yml`
   - **Environment**: (leave blank)

3. The GitHub workflow already has the required permissions:
   ```yaml
   permissions:
     contents: write
     id-token: write
   ```

### How It Works

When you push a tag (`v*`), GitHub Actions:
1. Requests a short-lived OIDC token from GitHub
2. Exchanges it for a PyPI upload token
3. Uploads the package
4. No secrets stored in GitHub!

### Manual Release (if needed)

```bash
# Build
python -m build

# Test
twine check dist/*

# Upload to PyPI (requires PyPI token configured locally)
twine upload dist/*
```

---

Happy releasing! 🐟
