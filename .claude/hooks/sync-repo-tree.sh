#!/usr/bin/env bash
# sync-repo-tree.sh — regenerates the auto-generated tree block inside
# REPO-TREE.md, between the AUTO-TREE:START / AUTO-TREE:END markers. Wired
# as a Stop hook by buildagent. Never touches anything outside those
# markers, never commits — just keeps the structural listing honest.
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$REPO_ROOT"

TREE_FILE="REPO-TREE.md"
[[ -f "$TREE_FILE" ]] || exit 0

command -v tree >/dev/null 2>&1 || exit 0

# BWK awk (macOS /usr/bin/awk) rejects a -v value containing embedded
# newlines ("newline in string"), so the tree output is passed via a temp
# file and read with getline instead of -v.
BLOCK_FILE="$(mktemp)"
trap 'rm -f "$BLOCK_FILE"' EXIT
tree -L 2 --gitignore -I '.git' > "$BLOCK_FILE" 2>/dev/null || true

TMP="$(mktemp)"
awk -v blockfile="$BLOCK_FILE" '
  BEGIN { skip = 0 }
  /<!-- AUTO-TREE:START -->/ {
    print
    print "```text"
    while ((getline line < blockfile) > 0) print line
    close(blockfile)
    print "```"
    skip = 1
    next
  }
  /<!-- AUTO-TREE:END -->/ { skip = 0 }
  skip { next }
  { print }
' "$TREE_FILE" > "$TMP"

if cmp -s "$TREE_FILE" "$TMP"; then
  rm -f "$TMP"
else
  mv "$TMP" "$TREE_FILE"
fi
