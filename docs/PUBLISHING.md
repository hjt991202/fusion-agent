# First commit and publication

This workspace already contained an empty Git repository; no reinitialization was needed.
The user authorized publishing to hjt991202, and the CLI identity was verified before publishing.
Use only reviewed synthetic data and source files.

For a new clone or a separate fresh project, the equivalent commands are:

```bash
git init
python .agents/skills/project-verification/scripts/verify.py
git add .
git diff --cached --stat
git commit -m "Build reproducible FusionAgent CPU workflow"
git branch -M main
gh auth status
gh repo create hjt991202/fusion-agent --public --source . --remote origin --push
```

Do not rerun repository creation when it already exists. Use `git push` for later reviewed commits.
Inspect `git diff --cached` for private images, secrets, weights and caches before each commit.
Git identity for this project uses the confirmed account's GitHub noreply address, not private email.
The initial public release contains offline SDK validation; live API access is not claimed.
