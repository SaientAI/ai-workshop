# Code signing policy

Free code signing provided by [SignPath.io](https://signpath.io/), certificate by [SignPath Foundation](https://signpath.org/).

Saient's Windows releases are built from the public source repositories by GitHub Actions. Signing through SignPath is currently pending project approval and workflow integration; until that is complete, Windows downloads are explicitly labelled as unsigned test builds.

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
