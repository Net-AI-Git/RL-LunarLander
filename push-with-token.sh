#!/usr/bin/env bash
# Push the **current branch** to GitHub using a Personal Access Token (prompted each run — not stored).
# Usage: ./push-with-token.sh
# Requires: git, checked-out branch (not detached HEAD), remote repo Net-AI-Git/RL-LunarLander

set -euo pipefail
cd "$(dirname "$0")"

REPO_PATH="Net-AI-Git/RL-LunarLander"
# GitHub user/org for HTTPS (same as repo owner).
GITHUB_USER="Net-AI-Git"

CURRENT_BRANCH="$(git branch --show-current || true)"
if [[ -z "${CURRENT_BRANCH}" ]]; then
  echo "Error: not on a named branch (detached HEAD). Run: git checkout <branch>" >&2
  exit 1
fi

echo "Pushing branch '${CURRENT_BRANCH}' to ${REPO_PATH} as ${GITHUB_USER} (token is hidden while typing)."
read -r -s -p "Personal access token: " TOKEN
echo ""

if [[ -z "${TOKEN}" ]]; then
  echo "Error: token is required." >&2
  exit 1
fi

# One-shot authenticated URL (token not written to disk). If your token has special
# characters (@, :, #, etc.), use a new token without them or URL-encode them.
AUTH_URL="https://${GITHUB_USER}:${TOKEN}@github.com/${REPO_PATH}.git"

TOKEN=""
unset TOKEN

git push "${AUTH_URL}" "HEAD:refs/heads/${CURRENT_BRANCH}"

echo "Done."
