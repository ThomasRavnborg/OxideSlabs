#!/usr/bin/env bash
set -e

echo "== Git status =="
git status --short

read -p "Commit changes? (y/n) " ans
[[ "$ans" == "y" ]] || exit 1

git add -A

if ! git diff --cached --quiet; then
    git commit -m "$(date +'%Y-%m-%d %H:%M:%S')"
else
    echo "No changes to commit."
fi

echo
echo "== Pushing to remotes =="
git push origin
git push cluster

echo
echo "== Done =="