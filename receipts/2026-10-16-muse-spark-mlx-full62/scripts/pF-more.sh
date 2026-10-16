#!/bin/bash
# By-hand continuation after the full-62 gateway was healthy: a second identical 64-token pass (determinism), then 128 and 256 tokens.
set -u
A="${RUN_ARTIFACT:?set RUN_ARTIFACT}"
GW=http://${STUDIO_IP}:18020
T=${CAPTURE_TOOLS}
PLACEMENT='M1 Ultra Mac Studio: layers 38-61 and lm_head (24 layers); 14-inch M2 Max: two ranks, layers 26-37 and 14-25 (12 + 12); 16-inch M1 Max: layers 2-13 (12); 13-inch M1: layers 0-1 and the embedding (2)'
echo "REPEAT64 start $(date -u +%T)"
CAPTURE_LENGTHS='64' bash "$T/capture-full62.sh" "$GW" "$A/capture-repeat64" local/muse-2bit 5 4 "$PLACEMENT" 2>&1 | tail -n 6
echo "LONG start $(date -u +%T)"
CAPTURE_LENGTHS='128 256' bash "$T/capture-full62.sh" "$GW" "$A/capture" local/muse-2bit 5 4 "$PLACEMENT" 2>&1 | tail -n 6
echo "MORE DONE $(date -u +%T)"
