#!/usr/bin/env bash
# Build the Top-1000 methodology pack on the Desktop: Straw_Hat_Coders_Methodology/{.md, .html, .pdf, diagram png},
# rebuild the final submission zip with the updated methodology, and commit.
# Usage (repo root): scripts/build_methodology_pack.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
OUT="$HOME/Desktop/Straw_Hat_Coders_Methodology"
mkdir -p "$OUT"

# 1. architecture diagram
dot -Tpng -Gdpi=180 docs/diagrams/final_architecture.dot -o docs/diagrams/final_architecture.png
cp docs/diagrams/final_architecture.png "$OUT/final_architecture.png"
cp docs/diagrams/final_architecture.png "$HOME/Desktop/BER_architecture.png"

# 2. methodology as Markdown (diagram referenced next to it)
cp docs/METHODOLOGY.md "$OUT/Straw_Hat_Coders_Methodology.md"

# 3. HTML with the diagram embedded, then PDF
.venv/bin/python -c "import markdown" 2>/dev/null || .venv/bin/python -m pip install -q markdown 2>/dev/null \
  || uv pip install -q --python .venv/bin/python markdown
.venv/bin/python - "$OUT" <<'PY'
import base64, sys
from pathlib import Path
import markdown
out = Path(sys.argv[1])
md = Path("docs/METHODOLOGY.md").read_text(encoding="utf-8")
img = base64.b64encode(Path("docs/diagrams/final_architecture.png").read_bytes()).decode()
body = markdown.markdown(md, extensions=["tables", "fenced_code", "sane_lists"])
fig = (f'<h2>Architecture</h2><p><img src="data:image/png;base64,{img}" '
       'style="width:100%;max-width:1000px"/></p>')
body = body.replace("<h2>Appendix</h2>", fig + "<h2>Appendix</h2>", 1)
css = """body{font-family:Helvetica,Arial,sans-serif;font-size:11pt;line-height:1.45;max-width:980px;margin:24px auto;
color:#111}h1{font-size:20pt}h2{font-size:15pt;border-bottom:1px solid #ccc;padding-bottom:3px;margin-top:22px}
h3{font-size:12.5pt}table{border-collapse:collapse;font-size:9.5pt;margin:8px 0}th,td{border:1px solid #bbb;
padding:3px 7px;text-align:left}th{background:#f0f0f0}code{font-size:9.5pt;background:#f5f5f5;padding:0 3px}
hr{border:0;border-top:1px solid #ddd}@page{size:A4;margin:14mm}"""
html = f'<!doctype html><html><head><meta charset="utf-8"><title>Straw Hat Coders - Methodology</title>' \
       f'<style>{css}</style></head><body>{body}</body></html>'
(out / "Straw_Hat_Coders_Methodology.html").write_text(html, encoding="utf-8")
print("html written")
PY
PDF="$OUT/Straw_Hat_Coders_Methodology.pdf"
HTML="$OUT/Straw_Hat_Coders_Methodology.html"
if command -v wkhtmltopdf >/dev/null; then wkhtmltopdf -q "$HTML" "$PDF" || true
else
  for b in chromium chromium-browser google-chrome google-chrome-stable microsoft-edge brave-browser; do
    if command -v "$b" >/dev/null; then "$b" --headless --disable-gpu --no-pdf-header-footer --print-to-pdf="$PDF" "file://$HTML" 2>/dev/null && break; fi
  done
fi
[ -s "$PDF" ] && echo "PDF: $PDF" || echo "No PDF tool found: open $HTML in a browser and Print -> Save as PDF"

# 4. final submission zip with the updated methodology, then commit (no attribution lines)
git add docs/METHODOLOGY.md docs/diagrams/final_architecture.dot docs/diagrams/final_architecture.png scripts/build_methodology_pack.sh
git commit -qm "Methodology aligned with the final submission #20; diagram updated; methodology pack builder" || true
mv "$HOME/Desktop/Straw_Hat_Coders_submission.zip" artifacts/old_zips/Straw_Hat_Coders_submission_20_prev2.zip 2>/dev/null || true
scripts/build_final_zip.sh submissions/20_20260927-2054 Straw_Hat_Coders
mv Straw_Hat_Coders_submission.zip "$HOME/Desktop/"
git push -q origin main || true
ls -la "$OUT"
