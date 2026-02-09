# Agents

## Skills

### fetch-action-logs

Fetch and display logs for a GitHub Actions run using the GitHub API.

**Location:** `.claude/skills/fetch-action-logs/SKILL.md`

**Usage:** Ask Claude to fetch CI logs or check a GitHub Actions run. Provide a run ID, or it will list recent runs and pick the latest.

**Requires:** `GITHUB_TOKEN` environment variable.

### fix-pr-comments

Read review comments on a GitHub PR and fix the issues raised.

**Location:** `.claude/skills/fix-pr-comments/SKILL.md`

**Usage:** Ask Claude to read and fix PR comments. Provide a PR number, or it will use the current branch's PR.
