#!/bin/bash
# Block D of the final-phase plan: the single held-out evaluation. N claim-based workers per GPU run
# `model/run_evaluation.py simulate --claim` (claims under output/model/claims/evaluation), so any number of lab
# PCs share the 3,220 networks; each refuses to start unless config/freeze.json holds and no listed file changed.
# Launched from the Mac, one kept-open session per machine:
#   nohup caffeinate -is ssh -o ControlPath=none -o ServerAliveInterval=60 <pc> 'bash /net/travail/mucandy/TRDP/scripts/cremi_evaluate.sh 3'
# When every machine's log says done: python model/run_evaluation.py score   (one machine, once)
N=${1:-3}; H=$(hostname -s); cd /net/travail/mucandy/TRDP || exit 1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTORCH_ALLOC_CONF=expandable_segments:True
L=/local/mucandy; mkdir -p $L
echo "$(date) $H evaluation start ($N workers, $(git rev-parse --short HEAD))" >> $L/evaluation.log
for w in $(seq 1 $N); do
  .venv/bin/python model/run_evaluation.py simulate --claim >> $L/evaluation_w$w.log 2>&1 &
  sleep 20
done
wait
echo "$(date) $H evaluation done" >> $L/evaluation.log
