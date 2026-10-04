#!/usr/bin/env bash
# Download datasets one at a time (chainlink E1). Safe to rerun: finished files are skipped.
# Usage: scripts/fetch_queue.sh [key ...]   (no keys = everything, in order)
# Progress: logs/queue.log (one line per dataset) and logs/fetch_<name>.log.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

# key | issue | source | id | name | extra args
QUEUE=$(cat <<'EOF'
mseg_probe|11|zenodo|15224240|myceliumseg|--only labeled-GL.zip unlabeled-GL2.zip
grazers|14|zenodo|14931953|fricker_grazers|
pod_thallus|16|figshare|22666138|podospora_thallus|
pod_m0|16|figshare|22140692|podospora_m0_raw|
pod_2019a|16|figshare|9897056|podospora_net2019a|
pod_2019b|16|figshare|10626968|podospora_net2019b|
networks|15|zenodo|5725751|fricker_networks|
aureo|17|zenodo|14001310|aureobasidium|
physarum|18|zenodo|22868368|physarum|--only README.md QC-NOTES.md manifests.zip CITATION.cff SHA256SUMS SHA256SUMS-release-documents release-summary.json physarum-001-dish0-r0c0.ome.zarr.zip
mseg_GL1|13|zenodo|15224240|myceliumseg|--only unlabeled-GL1.zip
mseg_GL3|13|zenodo|15224240|myceliumseg|--only unlabeled-GL3.zip
mseg_GL4|13|zenodo|15224240|myceliumseg|--only unlabeled-GL4.zip
mseg_GL5|13|zenodo|15224240|myceliumseg|--only unlabeled-GL5.zip
mseg_GL6|13|zenodo|15224240|myceliumseg|--only unlabeled-GL6.zip
mseg_GL7|13|zenodo|15224240|myceliumseg|--only unlabeled-GL7.zip
mseg_GSPOTS|13|zenodo|15224240|myceliumseg|--only unlabeled-GS_PO_TS.zip
EOF
)

# Wait for any fetch already running so two processes never write the same .part file.
while pgrep -f "scripts/fetch.py (zenodo|figshare) [0-9]+ [a-z_0-9]+" >/dev/null; do sleep 30; done

want=" ${*:-} "
while IFS='|' read -r key issue src id name extra; do
  [[ -n "${*:-}" && "$want" != *" $key "* ]] && continue
  echo "$(date -Is) start #$issue $key" >> logs/queue.log
  # Per-item cap: a stalled connection fails and can be resumed by rerunning.
  # shellcheck disable=SC2086
  if timeout 110m python3 scripts/fetch.py "$src" "$id" "$name" $extra >> "logs/fetch_$name.log" 2>&1; then
    echo "$(date -Is) done  #$issue $key" >> logs/queue.log
  else
    echo "$(date -Is) FAIL  #$issue $key (see logs/fetch_$name.log; rerun to resume)" >> logs/queue.log
  fi
done <<< "$QUEUE"
echo "$(date -Is) queue finished: ${*:-all}" >> logs/queue.log
