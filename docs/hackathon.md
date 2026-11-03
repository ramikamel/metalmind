# Event alignment and submission preparation

Verified November 2, 2026 against the public [event page](https://cerebralvalley.ai/e/meta-ai-developer-hackathon)
and its [Official Rules / Event Waiver](https://cerebralvalley.ai/e/meta-ai-developer-hackathon/waiver).
The waiver loaded in the browser after the text fetch showed only a loading state.

The event explicitly includes Muse Wonder. MetalMind's `muse-wonder-1.0` uses
that family through Meta Model API; this establishes model-family alignment,
not a separate eligibility certification. The individual-builder track and
Push the Frontier theme fit the user's stated solo software project.

The published submission deadline is November 4, 2026, 9 PM Eastern (8 PM
America/Chicago). The rules score impact 30%, Meta tooling 25%, technical execution
20%, and demo 25%; the demo criterion includes video quality. They permit building
on open-source components subject to their licenses and the stated ownership terms.
[Sources: event](https://cerebralvalley.ai/e/meta-ai-developer-hackathon),
[rules](https://cerebralvalley.ai/e/meta-ai-developer-hackathon/waiver).

The [official submission portal](https://cerebralvalley.ai/e/meta-ai-developer-hackathon/hackathon/submit)
redirected to sign-in. Its exact fields and video duration were not visible;
no submission, login, rule acceptance, or participant-eligibility attestation
was performed. These remain the entrant's portal steps.

## Prepared project material

- Public repository: https://github.com/ramikamel/metalmind
- Title: **MetalMind — verified autonomous Metal kernel optimization**.
- Proposed track/theme: individual-builder track / Push the Frontier.
- [Submission draft](submission-draft.md) maps the work to the rubric.
- [Demo guide](demo.md) and [three-minute silent video](../demo/video/metalmind-evidence-walkthrough.mp4)
  show saved generation evidence and a fresh correctness check. The presentation
  length was chosen for readability.
- [Function pilot](../results/2026-11-02-fp32-pilot/README.md) retains 2 wins and 7 no-win signatures.
- [Whole-module compiler comparison](../results/2026-11-02-module-baselines/README.md)
  explains why the stricter promotion gate kept the original module.
- `.env` and API credentials are excluded from the repository and demo.

## Dependency/source attribution

The controller, admission/guard logic, evaluation/report code, handwritten simple
Metal templates, and examples are project code. Muse Wonder generated the
candidate kernels identified by model/source provenance in the run evidence.
PyTorch supplies tensor semantics, MPS execution, compiler entry points and
framework baselines; NumPy supplies bootstrap array operations. Pydantic, PyYAML
and python-dotenv support contracts/configuration. Pytest and build are development
tools. Exact versions are in `requirements-dev.lock`; dependency licenses remain
with their distributions. This project does not claim ownership of those libraries.
