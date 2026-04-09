#!/usr/bin/env bash
# Set git user.name and user.email for **this repo only** (commit author).
# This does not log you into GitHub — use ./push-with-token.sh to cache HTTPS credentials.
#
# Usage: ./configure-git-identity.sh

set -euo pipefail
cd "$(dirname "$0")"

echo "Commit identity (shown on git log / GitHub). Not the same as GitHub login or PAT."
read -r -p "Name:  " GIT_NAME
read -r -p "Email: " GIT_EMAIL

if [[ -z "${GIT_NAME}" || -z "${GIT_EMAIL}" ]]; then
  echo "Error: name and email are required." >&2
  exit 1
fi

git config --local user.name "${GIT_NAME}"
git config --local user.email "${GIT_EMAIL}"

echo ""
echo "Configured (local):"
echo "  user.name  = $(git config --local --get user.name)"
echo "  user.email = $(git config --local --get user.email)"
