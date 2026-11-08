#!/bin/bash
# Four-machine run, 8 Nov 2026: the campaigns of this machine, plus two analysis workers that share
# the per-network work of all machines through claims (output/model/claims).
H=$(hostname -s)
cd /net/travail/mucandy/TRDP || exit 1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=/local/mucandy; mkdir -p $L
P=.venv/bin/python
SNN="h2_openclip_snn h2_dinov2_cls_snn h2_dinov2_patchmean_snn"
RATE="h2_openclip_rate h2_dinov2_cls_rate h2_dinov2_patchmean_rate"
ALL12="h0_dinov2_cls_snn h0_dinov2_patchmean_snn h0_openclip_snn h1_dinov2_cls_snn h1_dinov2_patchmean_snn h1_openclip_snn $SNN $RATE"
C=output/model/claims
CP="$P model/campaign.py --hypothesis h2 --representation openclip"

campaigns() {
  case $H in
    jessica)  $CP --recurrent --seeds 0-4 --stack 20 --finishers 2 >> $L/rec_campaign.log 2>&1
              $CP --variation nv512 --seeds 0-2 --stack 10 --finishers 2 > $L/robust_nv512.log 2>&1 ;;
    daniel)   $CP --recurrent --seeds 5-9 --stack 20 --finishers 2 > $L/rec_campaign.log 2>&1
              $CP --variation u6000 --seeds 0-2 --stack 20 --finishers 2 > $L/robust_u6000.log 2>&1 ;;
    vladimir) for v in nv128 lr3e-4 lr3e-3 tm10; do
                $CP --variation $v --seeds 0-2 --stack 20 --finishers 2 > $L/robust_$v.log 2>&1; done ;;
    cheonech) for v in tm40 ts2 ts10; do
                $CP --variation $v --seeds 0-2 --stack 20 --finishers 2 > $L/robust_$v.log 2>&1; done ;;
  esac
  echo "$(date) $H campaigns done" >> $L/parallel.log
}

analysis() {   # worker $1: ADM curves, then fit of c and Gate 2 checks of the 12 trained configurations
  w=$1
  $P model/extend_curves.py --population adm --claim >> $L/w${w}_adm.log 2>&1
  while [ "$(ls output/model/curve_extension_adm | wc -l)" -lt 2400 ]; do sleep 60; done
  mkdir -p $C
  if mkdir $C/adm_apply_1 2>/dev/null; then
    $P model/extend_curves.py --apply --population adm > $L/adm_apply.log 2>&1; touch $C/adm_apply_1/done
  else
    while [ ! -f $C/adm_apply_1/done ]; do sleep 30; done
  fi
  for pass in 1 2 3; do
    $P model/fit_gate.py --configurations $SNN --claim >> $L/w${w}_fit.log 2>&1
    $P model/gate2.py sweep --configurations $ALL12 --claim >> $L/w${w}_sweep.log 2>&1
    $P model/gate2.py predictions --claim >> $L/w${w}_pred.log 2>&1
    $P model/fit_gate.py --configurations $RATE --claim >> $L/w${w}_fit_rate.log 2>&1
  done
  echo "$(date) $H worker $w done" >> $L/parallel.log
}

echo "$(date) $H start" >> $L/parallel.log
campaigns &
analysis 1 &
sleep 20; analysis 2 &
wait
echo "$(date) $H ALL DONE" >> $L/parallel.log
