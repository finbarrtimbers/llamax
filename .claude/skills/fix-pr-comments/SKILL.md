---
name: fix-pr-comments
description: Read review comments on a GitHub PR and fix the issues raised. Use when the user asks to address, fix, or resolve PR comments or review feedback.
allowed-tools: Bash(gh pr view*), Bash(gh api*)
---

# Fix PR Comments

## Finding the PR

If no PR number is provided, find the current branch's PR:

```bash
gh pr view --json number,url -q '.number'
```

## Reading review comments

Get both general comments and inline review comments:

```bash
# General PR comments
gh pr view <PR_NUMBER> --json comments,reviews --jq '.comments[], .reviews[]'

# Inline review comments (on specific lines of code)
gh api repos/{owner}/{repo}/pulls/<PR_NUMBER>/comments
```

## Workflow

1. Fetch all comments using the commands above
2. Parse each comment to understand what change is being requested
3. Ignore bot comments that are purely informational (e.g. CI status)
4. For each actionable comment, make the requested fix in the codebase
5. Run the linter (`make style && make quality`) and tests (`uv run pytest`) after making changes
6. Commit and push the fixes
