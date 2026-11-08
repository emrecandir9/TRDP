#!/bin/bash
# Shared job list, 8 Nov 2026: every machine trains the remaining recurrent and robustness stacks,
# claimed one stack at a time (campaign.py --claim), so the eight GPUs never duplicate a network.
# A host first takes its own starting job, then the list in order (longest jobs first); two passes,
# so a stack whose process died on this host is taken over. Optional argument: worker number (log name),
# for a second worker on a GPU that one stack does not fill.
W=${1:-}; H=$(hostname -s); cd /net/travail/mucandy/TRDP || exit 1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTORCH_ALLOC_CONF=expandable_segments:True
L=/local/mucandy; mkdir -p $L; P=.venv/bin/python
JOBS=("--variation u6000 --seeds 0-2 --stack 20" "--recurrent --seeds 0-9 --stack 20"
      "--variation nv512 --seeds 0-2 --stack 10" "--variation nv128 --seeds 0-2 --stack 20"
      "--variation tm40 --seeds 0-2 --stack 20" "--variation lr3e-4 --seeds 0-2 --stack 20"
      "--variation lr3e-3 --seeds 0-2 --stack 20" "--variation tm10 --seeds 0-2 --stack 20"
      "--variation ts2 --seeds 0-2 --stack 20" "--variation ts10 --seeds 0-2 --stack 20")
case $H in
  jessica) s=1;; daniel) s=0;; vladimir) s=2;; cheonech) s=5;;
  moneo) s=0;; wensicia) s=1;; hayt) s=6;; albans) s=7;; *) s=0;;
esac
echo "$(date) $H work$W start" >> $L/parallel.log
for pass in 1 2; do
  for k in $s 0 1 2 3 4 5 6 7 8 9; do
    $P model/campaign.py --hypothesis h2 --representation openclip ${JOBS[$k]} --finishers 2 --claim >> $L/work$W.log 2>&1
  done
done
echo "$(date) $H work$W done" >> $L/parallel.log
