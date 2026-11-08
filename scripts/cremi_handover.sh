#!/bin/bash
# Handover, 8 Nov 2026: stops this host's old (claim-less) campaign so that the shared job list
# (cremi_work.sh) takes over. Health checks already running are claimed for their process, so no other
# machine repeats them.  Usage: cremi_handover.sh now
#                                cremi_handover.sh after <campaign log> <stack claim name>
# "after" first claims the stack being trained for the old process and stops it once that stack is done.
cd /net/travail/mucandy/TRDP || exit 1
C=output/model/claims; H=$(hostname); mkdir -p $C/finish $C/campaign
c=$(pgrep -u mucandy -f '^.venv/bin/python model/campaign.py' | head -1)
[ -z "$c" ] && { echo "no old campaign"; exit 0; }
claim_finishers() {
  for p in $(pgrep -u mucandy -f 'model/train.py --finish'); do
    n=$(basename "$(ps -o args= -p $p | awk '{print $NF}')" .trained.pt)
    mkdir $C/finish/$n 2>/dev/null && echo "$H $p" > $C/finish/$n/host
  done
}
kill -9 $(ps -o ppid= -p $c | tr -d ' ')     # the shell that would start the next old campaign
claim_finishers
if [ "$1" = after ]; then
  mkdir $C/campaign/$3 2>/dev/null && echo "$H $c" > $C/campaign/$3/host
  n0=$(grep -c stack_done $2)
  while [ "$(grep -c stack_done $2)" -le "$n0" ] && kill -0 $c 2>/dev/null; do sleep 20; done
  sleep 15                                    # the finished stack's first health checks start
fi
kill $c; sleep 5; claim_finishers
echo "$(date) $H old campaign $c stopped"
