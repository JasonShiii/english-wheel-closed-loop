# Dry run of 2026-10-29 (no sheet): flat scan -> zigzag A once (L->R) -> scan -> zigzag B twice (R->L, L->R), a scan after
# each pass. Scripted blocks: no planner, no controller decisions; everything measured and recorded under loops\<OutName>.
param([string]$OutName = "dry_two_zigzags_nopanel")
$ErrorActionPreference = "Continue"
$py = "C:\anaconda3\envs\EnglishWheel\python.exe"; $repo = "$HOME\english-wheel-closed-loop"; $scripts = "$HOME\Documents\GitHub\English-Wheel---DoBot-Files\Scripts"
Set-Location "$repo\real_testbed"
$out = "$repo\loops\$OutName"
if (Test-Path $out) { Remove-Item -Recurse -Force $out }; if (Test-Path "$out.log") { Remove-Item -Force "$out.log" }
$t0 = Get-Date
& $py -m ew_loop.cli rigloop --plant dobot --robot-scripts $scripts --auto-radii --script robot_dobot/sequence_dry_two_zigzags.json --allow-no-sheet `
    --target figures/rig/targets/S1_t042_c4.npz --out $out --calib "pc_post_processing/deliverable_092826/calibration results" `
    --clamp 12 --yes 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath "$out.log" |
    Where-Object { $_ -notmatch "UserWarning|plt.show|fig, \(ax_f|generic_filter|_nd_image|RuntimeWarning|^\s*$" }
"exit $LASTEXITCODE after {0:N0} s" -f ((Get-Date) - $t0).TotalSeconds
& $py -m ew_loop.trial_report $out
