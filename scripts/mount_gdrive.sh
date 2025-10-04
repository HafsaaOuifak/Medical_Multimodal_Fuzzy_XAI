#!/usr/bin/env bash
set -euo pipefail

# Configurable variables from .env
REMOTE="${RCLONE_REMOTE_NAME:-drive}"
SUBPATH="${RCLONE_DRIVE_SUBPATH:-}"
MOUNTPOINT="/workspace/gdrive"
SHARED="${RCLONE_SHARED_WITH_ME:-0}"

# Use the mounted rclone config
export RCLONE_CONFIG=/workspace/secrets/rclone.conf

# Create mountpoint
mkdir -p "$MOUNTPOINT"

# Create writable VFS/cache folder to avoid read-only errors
export RCLONE_CACHE_DIR=/workspace/data/rclone_cache
mkdir -p "$RCLONE_CACHE_DIR"

# Determine source remote path
if [[ -n "$SUBPATH" ]]; then
  SRC="${REMOTE}:${SUBPATH}"
else
  SRC="${REMOTE}:"
fi

echo "Mounting Google Drive: ${SRC} -> ${MOUNTPOINT}"
echo "Shared-with-me mode: ${SHARED}"
echo "Rclone cache directory: $RCLONE_CACHE_DIR"

EXTRA_FLAGS=()
if [[ "$SHARED" == "1" ]]; then
  EXTRA_FLAGS+=(--drive-shared-with-me)
fi

# Mount the folder with VFS cache
rclone mount "$SRC" "$MOUNTPOINT" \
  --vfs-cache-mode full \
  --vfs-cache-max-size 2G \
  --buffer-size 32M \
  --attr-timeout 1s \
  --dir-cache-time 30s \
  --poll-interval 30s \
  --umask 002 \
  --allow-other \
  --log-level INFO \
  --cache-dir "$RCLONE_CACHE_DIR" \
  "${EXTRA_FLAGS[@]}"
