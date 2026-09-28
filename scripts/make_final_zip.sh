#!/usr/bin/env bash
# Build ~/Desktop/Straw_Hat_Coders_submission.zip from the final submission #20 with the current methodology,
# verify it, and print READY. Usage: bash scripts/make_final_zip.sh
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
SUB=submissions/20_20260927-2054
Z="$HOME/Desktop/Straw_Hat_Coders_submission.zip"
fail() { echo "FAILED: $*"; exit 1; }

# 1. commit the methodology and diagram changes (the zip builder requires committed docs/code)
command -v dot >/dev/null && dot -Tpng -Gdpi=180 docs/diagrams/final_architecture.dot -o docs/diagrams/final_architecture.png
git add docs/METHODOLOGY.md docs/diagrams scripts
git diff --cached --quiet || git commit -qm "Methodology aligned with the final submission #20; diagram updated" || fail "git commit"
git diff --quiet HEAD -- code/ docs/METHODOLOGY.md || fail "uncommitted changes remain in code/ or docs/METHODOLOGY.md"

# 2. build the zip (old one kept in artifacts/old_zips/)
mkdir -p artifacts/old_zips
[ -f "$Z" ] && mv "$Z" "artifacts/old_zips/Straw_Hat_Coders_submission_$(date +%H%M%S).zip"
bash scripts/build_final_zip.sh "$SUB" Straw_Hat_Coders || fail "build_final_zip.sh"
mv Straw_Hat_Coders_submission.zip "$Z" || fail "move to Desktop"

# 3. verify
unzip -p "$Z" output/matching_results.tsv | cmp -s - "$SUB/matching_results.tsv" || fail "matching_results.tsv differs from #20"
unzip -p "$Z" output/candidate_pairs.tsv | cmp -s - "$SUB/candidate_pairs.tsv" || fail "candidate_pairs.tsv differs from #20"
unzip -p "$Z" Documentation_template.md | grep -q "for France" || fail "methodology in the zip is not the final version"
for f in code/business_entity_resolution/README.md code/business_entity_resolution/requirements.txt code/business_entity_resolution/src/ber/cli.py Documentation_template.md; do
  unzip -Z1 "$Z" | grep -qx "$f" || fail "missing $f"
done
n=$(unzip -p "$Z" -x "output/*" | grep -aci "claude\|anthropic"); [ "$n" = "0" ] || fail "$n attribution mentions in the zip"
git push -q origin main 2>/dev/null || echo "(git push skipped)"
ls -la "$Z"
echo "READY: $Z"
