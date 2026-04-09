#!/usr/bin/env bash
# Store GitHub HTTPS credentials in memory (git credential cache) so later `git push` / `git pull`
# work without embedding the token in the remote URL. Does not fetch, pull, or push.
# Usage: ./push-with-token.sh
# Token is prompted once per run; use when the cache expired or after reboot.

set -euo pipefail
cd "$(dirname "$0")"

GITHUB_USER="Net-AI-Git"

_origin="$(git remote get-url origin 2>/dev/null || true)"
if [[ -z "${_origin}" ]]; then
  echo "Error: no remote named origin." >&2
  exit 1
fi

if [[ "${_origin}" =~ ^git@github\.com: ]]; then
  echo "Error: origin uses SSH (${_origin}). This script stores HTTPS credentials." >&2
  echo "Use an HTTPS remote, e.g.: git remote set-url origin https://github.com/OWNER/REPO.git" >&2
  exit 1
fi

REPO_PATH=""
if [[ "${_origin}" =~ github\.com[:/]([^/]+)/([^/.]+) ]]; then
  REPO_PATH="${BASH_REMATCH[1]}/${BASH_REMATCH[2]}"
fi
unset _origin

# Memory-only cache for this repo (default 8h). Does not write the token to .git in clear text.
if ! git config --local --get credential.helper >/dev/null 2>&1; then
  git config --local credential.helper 'cache --timeout=28800'
fi

echo "GitHub HTTPS login for ${REPO_PATH:-github.com} as ${GITHUB_USER} (token hidden while typing)."
echo "Nothing will be fetched or pushed — only credentials are stored for later git commands."
read -r -s -p "Personal access token: " TOKEN
echo ""

if [[ -z "${TOKEN}" ]]; then
  echo "Error: token is required." >&2
  exit 1
fi

printf 'protocol=https\nhost=github.com\nusername=%s\npassword=%s\n\n' "${GITHUB_USER}" "${TOKEN}" | git credential approve

TOKEN=""
unset TOKEN

echo "Credentials cached in memory for this clone. When you want, run e.g.: git push origin <branch>"
