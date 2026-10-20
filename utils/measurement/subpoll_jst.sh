#!/bin/bash
# Read-only: logs every submission status change with a UTC timestamp (2-min resolution), for up to 30 h.
cd /import/fox/will/graduation_thesis/RSNA_kaggle
declare -A last
for n in $(seq 1 900); do
  while IFS= read -r line; do
    ref=$(echo "$line" | awk '{print $1}'); [[ "$ref" =~ ^[0-9]+$ ]] || continue
    st=$(echo "$line" | grep -oE "SubmissionStatus\.[A-Z]+"); sc=$(echo "$line" | grep -oE "SubmissionStatus\.[A-Z]+ +[0-9.]+" | awk '{print $2}')
    desc=$(echo "$line" | grep -oE "Notebook [^|]+\| Version [0-9]+")
    key="$st $sc"
    if [ "${last[$ref]}" != "$key" ]; then echo "$(TZ=Asia/Tokyo date '+%Y-%m-%d %H:%M') JST | $ref | $desc | $st ${sc:-}"; last[$ref]="$key"; fi
  done < <(kaggle competitions submissions rsna-knee-abnormality-detection 2>/dev/null | head -12)
  sleep 120
done
