#!/bin/bash
# 10 Nov 2026: after the all-context campaign, its ADM curves and fit of c; then the 0.5-ms subset with
# its curves and fit (docs/model-method.md 6.21, 6.23). Every step skips finished outputs.
cd /net/travail/mucandy/TRDP || exit 1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTORCH_ALLOC_CONF=expandable_segments:True
P=.venv/bin/python; L=/local/mucandy; mkdir -p $L
say() { echo "$(date +%H:%M:%S) $*" >> $L/review_queue.log; }
say "waiting for the all-context campaign"
while pgrep -u mucandy -f '^.venv/bin/python model/campaign.py' > /dev/null; do sleep 60; done
curves_and_fit() {
  $P model/extend_curves.py --population adm >> $L/review_steps.log 2>&1
  $P model/extend_curves.py --apply --population adm > $L/review_apply.log 2>&1
  $P model/fit_gate.py --configurations "$1" >> $L/review_steps.log 2>&1
  say "$1: $(ls output/model/campaign/$1/health_*.json 2>/dev/null | wc -l) reports, $(ls output/model/gate_fit/$1 2>/dev/null | wc -l) fits"
}
curves_and_fit h2_openclip_snn_allctx
say "training the 0.5-ms subset"
$P model/campaign.py --hypothesis h2 --representation openclip --variation dt0.5 --seeds 0 --stack 20 --finishers 2 > $L/dt05.log 2>&1
curves_and_fit h2_openclip_snn_dt0.5
$P model/fit_gate.py --summary --configurations h2_openclip_snn h2_openclip_snn_allctx h2_openclip_snn_dt0.5 > $L/review_summary.log 2>&1
say "done"
