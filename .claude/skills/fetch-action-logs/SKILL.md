---
name: fetch-action-logs
description: Fetch and display logs for a GitHub Actions run. Use when the user asks to get, check, or view CI logs for a GitHub Actions run.
allowed-tools: Bash(curl*), Bash(unzip*), Bash(gh run list*), Bash(gh repo view*)
---

# Fetch GitHub Actions Run Logs

## Finding the run ID

If no run ID is provided, list recent runs:

```bash
gh run list --limit 5
```

The run ID is the numeric ID in the output (second to last column).

## Downloading logs

```bash
REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)

curl -L \
  -H "Authorization: Bearer $GITHUB_TOKEN" \
  -H "Accept: application/vnd.github+json" \
  "https://api.github.com/repos/$REPO/actions/runs/<RUN_ID>/logs" \
  -o /tmp/gh_logs.zip
```

## Extracting and displaying logs

```bash
rm -rf /tmp/gh_logs && unzip -o /tmp/gh_logs.zip -d /tmp/gh_logs/
```

Then read the relevant log files from `/tmp/gh_logs/` to show the user what happened.

## Getting logs for a specific job

```bash
curl -L \
  -H "Authorization: Bearer $GITHUB_TOKEN" \
  -H "Accept: application/vnd.github+json" \
  "https://api.github.com/repos/$REPO/actions/jobs/<JOB_ID>/logs" \
  -o /tmp/gh_job.log
```

The job endpoint returns plain text instead of a zip.
