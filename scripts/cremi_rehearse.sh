#!/bin/bash
# Task A3 of the final-phase plan: the training-source rehearsal of the evaluation on every network,
# N claim-based workers per GPU (scripts/rehearse.py; claims under output/model/claims/rehearsal), so several
# machines can share the job list. Launched from the Mac in a kept-open session:
#   nohup caffeinate -is ssh -o ControlPath=none -o ServerAliveInterval=60 joe 'bash /net/travail/mucandy/TRDP/scripts/cremi_rehearse.sh 3'
N=${1:-3}; H=$(hostname -s); cd /net/travail/mucandy/TRDP || exit 1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTORCH_ALLOC_CONF=expandable_segments:True
L=/local/mucandy; mkdir -p $L
echo "$(date) $H rehearsal start ($N workers)" >> $L/rehearsal.log
for w in $(seq 1 $N); do
  .venv/bin/python scripts/rehearse.py simulate --claim >> $L/rehearsal_w$w.log 2>&1 &
  sleep 20
done
wait
echo "$(date) $H rehearsal done" >> $L/rehearsal.log
