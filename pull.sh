#!/usr/bin/env bash
set -e

CLUSTER="s214434@sylg.fysik.dtu.dk"
REMOTE_DIR="~/OxideSlabs"

echo "== Syncing results from cluster =="
rsync -av --progress \
  "$CLUSTER:$REMOTE_DIR/results/" \
  results/
