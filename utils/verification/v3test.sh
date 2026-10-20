#!/bin/bash
# Local v3 verification on a 60-study fake test root. Every v3 variant must reproduce v1's submission exactly.
R=/import/fox/will/graduation_thesis/RSNA_kaggle; S=/tmp/claude-200134/-import-fox-will-graduation-thesis/7915c735-abec-4f51-9668-8df76b65e1cf/scratchpad
PY=/home/gp.sc.cc.tohoku.ac.jp/wanrencheng/anaconda3/envs/rsna/bin/python
mk() {  # $1 kernel file, $2 local copy, $3 output csv
  python3 - "$R" "$S" "$1" "$2" "$3" <<'PY'
import sys
R, S, ker, out_py, out_csv = sys.argv[1:6]
s = open(ker).read()
for a, b in [('    cands = ["/kaggle/input/competitions/rsna-knee-abnormality-detection",\n',
              f'    cands = ["{S}/fake_root", "/kaggle/input/competitions/rsna-knee-abnormality-detection",\n'),
             ('    direct = [f"/kaggle/input/raptor-knee-arms/{fname}",\n',
              f'    direct = [f"{R}/dread/train/{{fname}}", f"/kaggle/input/raptor-knee-arms/{{fname}}",\n'),
             ('    out = "/kaggle/working/submission.csv"\n', f'    out = "{out_csv}"\n')]:
    assert s.count(a) == 1, a; s = s.replace(a, b)
open(out_py, "w").write(s)
PY
}
mk $R/kaggle_kernel_dreadEff/rsna_dread_eff_d96_k43.py $S/t_v1.py $S/sub_T0.csv
for t in T1 T2 T3 T4; do mk $R/kaggle_kernel_dreadEff3/rsna_dread_eff_d96_k43_2gpu.py $S/t_v3_$t.py $S/sub_$t.csv; done
cd $S
run() { echo "=== $1"; shift; ( "$@" ) 2>&1 | grep -E "gpus|self-check|retired|FALLBACK|stalled|drain|pool\]|DONE|Error|Traceback" | cut -c1-150; }
run "T0 v1 reference" taskset -c 0-3 $PY t_v1.py
run "T1 v3, natural (1 GPU here)" taskset -c 0-3 $PY t_v3_T1.py
run "T2 v3, two replicas on cuda:0" env EFF_FORCE_DEVS=cuda:0,cuda:0 taskset -c 0-3 $PY t_v3_T2.py
run "T3 v3, replica 1 fails" env EFF_FORCE_DEVS=cuda:0,cuda:0 EFF_TEST_FAIL_DEV=1 taskset -c 0-3 $PY t_v3_T3.py
run "T4 v3, replica 0 fails" env EFF_FORCE_DEVS=cuda:0,cuda:0 EFF_TEST_FAIL_DEV=0 taskset -c 0-3 $PY t_v3_T4.py
$PY - <<'PY'
import pandas as pd, numpy as np
ref = pd.read_csv("sub_T0.csv")
assert len(ref) == 60
for t in ["T1", "T2", "T3", "T4"]:
    x = pd.read_csv(f"sub_{t}.csv")
    same_ids = (x.StudyInstanceUID == ref.StudyInstanceUID).all()
    d = np.abs(x.iloc[:, 1:].values - ref.iloc[:, 1:].values).max()
    n05 = int((x.iloc[:, 1:].values == 0.5).all(1).sum())
    print(f"{t}: rows {len(x)} ids aligned {same_ids} | max|d| vs v1 {d:.2e} | identical {d == 0} | all-0.5 rows {n05}")
PY
