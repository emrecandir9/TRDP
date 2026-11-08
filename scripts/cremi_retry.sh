#!/bin/bash
# Retry worker, 8 Nov 2026: removes this host's claims whose process died (out-of-memory), then runs
# the fit of c, the control predictions and the twin fits again, until every network has its result.
H=$(hostname -s); cd /net/travail/mucandy/TRDP || exit 1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTORCH_ALLOC_CONF=expandable_segments:True
L=/local/mucandy; P=.venv/bin/python
SNN="h2_openclip_snn h2_dinov2_cls_snn h2_dinov2_patchmean_snn"
RATE="h2_openclip_rate h2_dinov2_cls_rate h2_dinov2_patchmean_rate"
echo "$(date) $H retry start" >> $L/parallel.log
while true; do
  $P ~/trdp_runs/stale.py --clean >> $L/retry.log 2>&1
  $P model/fit_gate.py --configurations $SNN --claim >> $L/retry.log 2>&1
  $P model/gate2.py predictions --claim >> $L/retry.log 2>&1
  $P model/fit_gate.py --configurations $RATE --claim >> $L/retry.log 2>&1
  fits=$(ls output/model/gate_fit/h2_*/ | grep -c json); preds=$(find output/model/gate2/predictions -name '*.json' | wc -l)
  [ "$fits" -ge 1200 ] && [ "$preds" -ge 1800 ] && break
  sleep 120
done
echo "$(date) $H retry done (fits $fits, predictions $preds)" >> $L/parallel.log
