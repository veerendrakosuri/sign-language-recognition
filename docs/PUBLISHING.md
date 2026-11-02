# Publishing to PyPI

PyPI publication uses OpenID Connect trusted publishing, so the repository stores no API token. The workflow is manual and publishes an existing immutable release tag only after rebuilding and checking its distributions.

One-time account setup:

1. Create or sign in to the owner's PyPI account and enable two-factor authentication.
2. In PyPI account **Publishing**, add a pending GitHub publisher for project `signlab-asl`.
3. Enter owner `veerendrakosuri`, repository `sign-language-recognition`, workflow `publish-pypi.yml`, and environment `pypi`.
4. In the GitHub repository, create an environment named `pypi`. Add required reviewers if the account plan offers them.

To publish, open **Actions → Publish to PyPI → Run workflow**, enter an existing tag such as `v0.5.0`, and select the confirmation box. The workflow verifies that the tag exactly matches the package version before requesting a short-lived PyPI credential. PyPI versions cannot be replaced; publish a new version if an archive needs correction.

A pending publisher does not reserve the project name. Verify the project page immediately before the first publication. After a successful upload, use the normal `pip install signlab-asl` command.
