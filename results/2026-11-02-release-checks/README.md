# Function-level development release checks

Implementation commit: `13e3af6f5c2e0a7c3e04fa808db84deb7797964e`.
The enclosing documentation commit adds the public evidence and this manifest.

- Complete local CPU/GPU suite: **88 passed**, no skips; Metal health passed.
- Wheel and sdist built; final wheel installed in a fresh virtual environment
  outside the checkout, with no API key or `PYTHONPATH`.
- Installed package import resolved inside the fresh environment; the console
  entry point passed a real Metal health probe and a no-request campaign preview.
- Every curated artifact's source/evidence identity was checked before publication.
- Exact local-key scan passed; `.env` is ignored and excluded from Git.
- HTML content and escaping tests passed. Browser visual preview was unavailable
  because its policy blocks local file URLs; no alternate route was attempted.

`release.json` records the implementation revision, source tree digest, hardware,
model/request limits, dependency/wheel hashes, artifact IDs, and remaining work.
`clean-install.json` retains the isolated installation result. The full campaign,
graph extensions, and event eligibility remain outside this checkpoint.
