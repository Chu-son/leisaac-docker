#!/bin/bash
# レポート (sweep_report.md) を PDF にする。ホストで実行する (uvx と Google Chrome を使う)。
#   docs/report/build_pdf.sh
set -euo pipefail
cd "$(dirname "$0")"
python3 make_figures.py
body=$(uvx --quiet --from markdown==3.7 markdown_py -x tables -x fenced_code -x sane_lists sweep_report.md)
cat > sweep_report.html <<EOF
<!DOCTYPE html>
<html lang="ja"><head><meta charset="utf-8"><title>LeKiwi の箒掃除タスク</title>
<style>
@page { size: A4; margin: 18mm 16mm; }
body { font-family: "Noto Sans CJK JP", sans-serif; font-size: 10pt; line-height: 1.65; color: #0b0b0b; }
h1 { font-size: 17pt; border-bottom: 2px solid #2a78d6; padding-bottom: 4px; }
h2 { font-size: 13pt; margin-top: 1.4em; border-left: 4px solid #2a78d6; padding-left: 8px; break-after: avoid; }
h3 { font-size: 11pt; margin-top: 1.1em; break-after: avoid; }
table { border-collapse: collapse; margin: 0.6em 0; font-size: 9pt; break-inside: avoid; }
th, td { border: 1px solid #c3c2b7; padding: 3px 7px; vertical-align: top; }
th { background: #f0efec; }
code { font-family: "Noto Sans Mono CJK JP", monospace; font-size: 0.9em; background: #f3f3f0; padding: 0 3px; }
pre { background: #f3f3f0; padding: 8px; font-size: 8.5pt; break-inside: avoid; }
img { max-width: 100%; display: block; margin: 0.5em auto; }
p:has(> img) { break-inside: avoid; break-after: avoid; }
p:has(> em:only-child) { text-align: center; font-size: 8.5pt; color: #52514e; margin-top: 0; }
</style></head><body>
$body
</body></html>
EOF
google-chrome --headless=new --disable-gpu --no-pdf-header-footer --print-to-pdf="$PWD/sweep_report.pdf" "file://$PWD/sweep_report.html" 2>/dev/null
rm sweep_report.html
echo "-> $PWD/sweep_report.pdf"
