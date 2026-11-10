#!/usr/bin/env bash
# Push this repo to a Hugging Face Docker Space. Usage: HF_TOKEN=... ./scripts/deploy_hf.sh <user>/<space>
set -euo pipefail
SPACE="$1"; TMP=$(mktemp -d)
git archive HEAD | tar -x -C "$TMP"
cd "$TMP"
{ printf -- '---\ntitle: LitScout\nemoji: 🔎\ncolorFrom: indigo\ncolorTo: purple\nsdk: docker\napp_port: 7860\npinned: false\nshort_description: AI research agent for verified literature reviews\n---\n\n'; cat README.md; } > README.hf && mv README.hf README.md
git init -q && git checkout -q -b main && git add -A && git -c user.name=deploy -c user.email=deploy@users.noreply.github.com commit -qm "Deploy LitScout"
git push -q --force "https://user:${HF_TOKEN}@huggingface.co/spaces/${SPACE}" main
echo "pushed to https://huggingface.co/spaces/${SPACE}"
