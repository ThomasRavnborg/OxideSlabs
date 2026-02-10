#!/usr/bin/env bash
set -e

CLUSTER="s214434@sylg.fysik.dtu.dk"
REMOTE_DIR="~/OxideSlabs"

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
read -p "Pull on cluster now? (y/n) " pull_ans
[[ "$pull_ans" == "y" ]] || exit 0

echo "== Pulling on cluster =="
ssh "$CLUSTER" << EOF
cd $REMOTE_DIR
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "Cluster repo has uncommitted changes. Aborting."
    exit 1
fi
git pull
EOF

echo
echo "== Done =="
