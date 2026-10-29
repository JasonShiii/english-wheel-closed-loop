# Dry run with the planner (2026-10-29, no sheet): the robot runs and scans for real; the planner (visfix model on ideas2
# through the tunnel, localhost:8765) and the controller see the simulated trajectory behind target S1_t042 instead of
# the scans (--fake-fields). Everything recorded live under loops\<OutName>.
param([string]$OutName = "dry_planner_fake_S1_t042")
$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"; $env:PYTHONUTF8 = "1"; [Console]::OutputEncoding = [Text.Encoding]::UTF8
$py = "C:\anaconda3\envs\EnglishWheel\python.exe"; $repo = "$HOME\english-wheel-closed-loop"; $scripts = "$HOME\Documents\GitHub\English-Wheel---DoBot-Files\Scripts"
Set-Location "$repo\real_testbed"
$out = "$repo\loops\$OutName"
if (Test-Path $out) { Remove-Item -Recurse -Force $out }; if (Test-Path "$out.log") { Remove-Item -Force "$out.log" }
$t0 = Get-Date
& $py -m ew_loop.cli rigloop --plant dobot --robot-scripts $scripts --auto-radii --server http://localhost:8765 `
    --fake-fields figures/rig/fake_fields/S1_t042_sim --allow-no-sheet `
    --target figures/rig/targets/S1_t042_c4.npz --out $out --calib "pc_post_processing/deliverable_092826/calibration results" `
    --clamp 12 --n 3 --max-passes 8 --yes 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath "$out.log" |
    Where-Object { $_ -notmatch "UserWarning|plt.show|fig, \(ax_f|generic_filter|_nd_image|RuntimeWarning|^\s*$|Pulses: |Wheel pulse " }
"exit $LASTEXITCODE after {0:N0} s" -f ((Get-Date) - $t0).TotalSeconds
& $py -m ew_loop.trial_report $out
