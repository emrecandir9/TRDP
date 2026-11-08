#!/bin/bash
# Follow-up analysis, 9 Nov 2026, alongside the campaigns: ADM curves of newly finished networks,
# applied by one elected machine, then the fit of c for the robustness folders (and the recurrent
# folder once ~/trdp_runs/rec_mode_ok exists, i.e. its mode check passed, docs/model-method.md §6.20).
# Loops until every new network has its fit.
H=$(hostname -s); cd /net/travail/mucandy/TRDP || exit 1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTORCH_ALLOC_CONF=expandable_segments:True
L=/local/mucandy; mkdir -p $L; P=.venv/bin/python; C=output/model/claims
VAR=""; for v in nv128 nv512 lr3e-4 lr3e-3 u6000 tm10 tm40 ts2 ts10; do VAR="$VAR h2_openclip_snn_$v"; done
echo "$(date) $H phase2 start" >> $L/parallel.log
while true; do
  FITS=$VAR; want=540
  [ -f ~/trdp_runs/rec_mode_ok ] && { FITS="$VAR h2_openclip_snn_rec"; want=740; }
  $P model/extend_curves.py --population adm --claim >> $L/phase2.log 2>&1
  find $C -maxdepth 1 -name adm_apply_lock -mmin +30 -exec rmdir {} \; 2>/dev/null
  if mkdir $C/adm_apply_lock 2>/dev/null; then
    n=$(ls output/model/curve_extension_adm | wc -l); last=$(cat $C/adm_applied 2>/dev/null || echo 0)
    if [ "$n" -gt "$last" ]; then
      $P model/extend_curves.py --apply --population adm > $L/phase2_apply.log 2>&1 && echo $n > $C/adm_applied
    fi
    rmdir $C/adm_apply_lock
  fi
  $P model/fit_gate.py --configurations $FITS --claim >> $L/phase2.log 2>&1
  have=0; for c in $FITS; do have=$((have + $(ls output/model/gate_fit/$c 2>/dev/null | wc -l))); done
  [ "$have" -ge "$want" ] && [ "$want" -eq 740 ] && break
  sleep 300
done
echo "$(date) $H phase2 done ($have fits)" >> $L/parallel.log
