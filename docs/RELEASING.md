# Releasing

Package: `inference-control-plane` on PyPI.
Import package: `inference_control_plane`. Single version source: `version` in
`pyproject.toml`; `__version__` is derived from package metadata at runtime.

Publishing uses GitHub Trusted Publishing (OIDC) — no PyPI API token is
stored in the repository. The workflow is
`.github/workflows/publish.yml`: pushing a `v*` tag publishes to PyPI;
`workflow_dispatch` can publish to TestPyPI (default) or PyPI manually.

> Repository: `https://github.com/yavuzhansarmaz/inference-control-plane`.
> The trusted-publisher settings below use this owner/repo.

## One-time manual setup (maintainer)

1. Create a PyPI account at https://pypi.org/account/register/ and enable
   2FA. Repeat on https://test.pypi.org if you will use TestPyPI.
2. Create the GitHub repository assumed above
   (`yavuzhansarmaz/inference-control-plane`), push this code, and create
   two GitHub Environments named `pypi` and `testpypi` (Settings →
   Environments). Optionally add required reviewers on `pypi`.
3. Register a *pending publisher* on PyPI before the first release:
   PyPI → Your account → Publishing → "Add a new pending publisher":
   - PyPI project name: `inference-control-plane`
   - Owner: `yavuzhansarmaz` · Repository: `inference-control-plane`
   - Workflow filename: `publish.yml` · Environment name: `pypi`
   Do the same on TestPyPI with environment name `testpypi`.
   A pending publisher lets the first OIDC publish create the project;
   no token is issued or stored.
4. Verify the package name is still free before the first publish
   (`https://pypi.org/pypi/inference-control-plane/json` should
   return 404 until you publish).

## Release steps

1. Bump `version` in `pyproject.toml` only, add a CHANGELOG
   entry, run `python -m pytest` and `python -m ruff check src tests`.
2. Optional dry run: run the *Publish* workflow via `workflow_dispatch`
   with target `testpypi`, then check
   https://test.pypi.org/project/inference-control-plane/ and
   install from TestPyPI in a clean venv.
3. Tag and push (maintainer pushes manually):
   `git tag vX.Y.Z && git push origin vX.Y.Z`. The tag triggers the
   `pypi` job, which builds, runs `twine check`, and publishes via OIDC.
4. Verify on https://pypi.org/project/inference-control-plane/ and
   smoke-test: `pip install inference-control-plane==X.Y.Z` in a
   clean venv, then
   `python -c "from inference_control_plane import VllmClient, __version__"`.
5. In each consuming host application: bump the pinned
   `inference-control-plane==X.Y.Z`, run its regression tests,
   deploy.

Local build/check without publishing:

```bash
python -m pip install build twine
python -m build
python -m twine check dist/*
```

Rollback: re-pin the previous version in the host application and redeploy. The
package owns no database schema, so no data rollback is needed. A bad
PyPI release cannot be overwritten — yank it on PyPI and publish a new
patch version instead.
