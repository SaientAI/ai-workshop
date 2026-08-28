# Code signing policy

Free code signing provided by [SignPath.io](https://signpath.io/), certificate by [SignPath Foundation](https://signpath.org/).

Saient's Windows releases are built from the public source repositories by GitHub Actions. The release workflow submits the NSIS installer to SignPath for signing; signing is currently pending SignPath project approval, and until the credentials below exist the workflow skips the signing steps and Windows downloads remain explicitly labelled as unsigned test builds.

### Enabling signing

The `windows` job in `.github/workflows/release.yml` signs only when `SIGNPATH_API_TOKEN` is present, so nothing needs to change in the workflow once the project is approved — set these on the `SaientAI/ai-workshop` repository:

| Kind | Name |
|---|---|
| Secret | `SIGNPATH_API_TOKEN` |
| Secret | `SIGNPATH_ORGANIZATION_ID` |
| Variable | `SIGNPATH_PROJECT_SLUG` |
| Variable | `SIGNPATH_SIGNING_POLICY_SLUG` |
| Variable | `SIGNPATH_ARTIFACT_CONFIGURATION_SLUG` |

The three slugs must match the project, signing policy, and artifact configuration created in SignPath. The unsigned installer is uploaded as the `internal-windows-unsigned` artifact for SignPath to consume; the draft release only collects `saient-*`, so the unsigned copy is never published alongside the signed one.

Note that Authenticode signing stops Windows reporting the installer as from an unknown publisher, but SmartScreen reputation is earned separately over download history — expect warnings to persist for a while after the first signed release.

## Source and builds

- Application source: [SaientAI/ai-workshop](https://github.com/SaientAI/ai-workshop)
- Inference-engine source: [SaientAI/saient-quartz](https://github.com/SaientAI/saient-quartz)
- Automated build system: [GitHub Actions](https://github.com/SaientAI/ai-workshop/actions)
- Releases and checksums: [saient.co.uk](https://saient.co.uk/#download)

Release artifacts submitted for signing must be produced by the repository's automated Windows release workflow from the revision identified by the signing request. Every signing request requires manual approval.

## Team roles

- Committer and reviewer: [Chris Hall (@SaientAI)](https://github.com/SaientAI)
- Signing approver: [Chris Hall (@SaientAI)](https://github.com/SaientAI)

Contributions from people without direct commit access require review before they are merged. Repository and SignPath accounts used by the project must have multi-factor authentication enabled.

## Privacy

Saient's [privacy policy](https://saient.co.uk/privacy) describes its local-first processing and the limited data handled by the project website. The desktop application does not transfer prompts, models, or generated outputs unless the user explicitly requests an operation that requires network access.
