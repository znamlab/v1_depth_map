#!/bin/bash
#SBATCH --job-name=figshare_up
#SBATCH --partition=ncpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=48:00:00
#SBATCH --output=/camp/home/blota/slurm_logs/figshare_%j.out

# Upload the v1_depth_map data package to figshare over FTPS.
#
#   sbatch v1_depth_map/precompute_data/figshare_upload.sbatch
#
# Credentials are read from ~/.netrc (chmod 600), one line:
#   machine ftps.figshare.com login <figshare ftp user> password <figshare ftp password>
#
# How the figshare FTP uploader behaves (figshare user guide, "Upload large
# files and bulk upload using FTPS or the API"):
#   - files must go in a folder inside data/; the folder name becomes the title
#     of a NEW item (an existing item cannot be updated this way);
#   - no sub-folders; 5 TB limit per file;
#   - each file is processed as soon as it is uploaded, then deleted from the
#     FTP folder and recorded in processed_files_log.txt in that folder.
#
# Because processed files disappear from the server, the files already sent
# are recorded in $DONE_LIST and skipped when the script is resubmitted.
# Delete that file to start again from scratch.

set -uo pipefail

DATA_DIR=/nemo/lab/znamenskiyp/home/shared/published_data/v1_depth_map/science_resubmission_upload
# Folder created under data/ on the figshare FTP server; figshare turns each
# folder there into one draft item named after it.
REMOTE_DIR=He_et_al_2024
FTP_URL="ftp://ftps.figshare.com/data/${REMOTE_DIR}"
MAX_TRIES=5
# With TLS 1.3, curl 7.61 (NEMO) fails on every file above 64 KiB:
# "426 Failure reading network stream". TLS 1.2 works.
CURL_TLS=(--tls-max 1.2)
DONE_LIST=/camp/home/blota/slurm_logs/figshare_uploaded_${REMOTE_DIR}.txt

# Small files first, the 243 GB archive last.
FILES=(
  README.md
  config_template.yml
  ARCHIVES.md5
  MANIFEST_raw.tsv
  MANIFEST_processed.tsv
  MANIFEST_analysis.tsv
  offline_database.json
  analysis.zip
  raw.zip
  processed.zip
)

cd "$DATA_DIR" || { echo "Cannot enter $DATA_DIR"; exit 1; }

# Stop before sending anything if the folder is not exactly the expected set.
for f in "${FILES[@]}"; do
  [ -f "$f" ] || { echo "MISSING: $f"; exit 1; }
done
n_found=$(find . -mindepth 1 -maxdepth 1 | wc -l)
if [ "$n_found" -ne "${#FILES[@]}" ]; then
  echo "Expected ${#FILES[@]} files in $DATA_DIR, found $n_found:"
  ls -la
  exit 1
fi

touch "$DONE_LIST"
failed=()
for f in "${FILES[@]}"; do
  if grep -qxF "$f" "$DONE_LIST"; then
    echo "$(date '+%F %T') skipping $f (listed in $DONE_LIST)"
    continue
  fi
  size=$(stat -c %s "$f")
  ok=0
  for try in $(seq 1 "$MAX_TRIES"); do
    echo "$(date '+%F %T') uploading $f ($size bytes), attempt $try/$MAX_TRIES"
    if curl "${CURL_TLS[@]}" --ssl-reqd --netrc --ftp-pasv --ftp-create-dirs \
        --silent --show-error --connect-timeout 60 \
        --speed-limit 1024 --speed-time 600 \
        -C - -T "$f" "${FTP_URL}/"; then
      ok=1
      break
    fi
    # -C - resumes if the partial file is still on the server. If figshare
    # already processed the partial file, the item holds a truncated copy to
    # delete by hand in My data.
    echo "$(date '+%F %T') attempt $try failed for $f: check the item for a truncated copy"
    sleep 60
  done
  if [ "$ok" -eq 1 ]; then
    echo "$f" >> "$DONE_LIST"
    echo "$(date '+%F %T') done $f"
  else
    echo "$(date '+%F %T') FAILED: $f"
    failed+=("$f")
  fi
done

echo
echo "Remote folder (files still listed here are not processed yet):"
curl "${CURL_TLS[@]}" --ssl-reqd --netrc --ftp-pasv --silent --show-error "${FTP_URL}/"
echo
echo "processed_files_log.txt:"
curl "${CURL_TLS[@]}" --ssl-reqd --netrc --ftp-pasv --silent --show-error "${FTP_URL}/processed_files_log.txt" \
  || echo "(no processing log yet)"

if [ "${#failed[@]}" -gt 0 ]; then
  echo "FAILED FILES: ${failed[*]}"
  exit 1
fi
echo "All ${#FILES[@]} files uploaded. Check sizes and md5 of the files in the figshare item."
