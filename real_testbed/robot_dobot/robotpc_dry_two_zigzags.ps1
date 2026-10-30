# Dry run of 2026-10-29 (no sheet): flat scan -> zigzag A once (L->R) -> scan -> zigzag B twice (R->L, L->R), a scan after
# each pass. Scripted blocks: no planner, no controller decisions; everything measured and recorded under loops\<OutName>.
param([string]$OutName = "dry_two_zigzags_nopanel", [string]$RobotCode = "", [switch]$AllowUnverifiedRobotCode)
$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"; $env:PYTHONUTF8 = "1"; [Console]::OutputEncoding = [Text.Encoding]::UTF8
$py = "C:\anaconda3\envs\EnglishWheel\python.exe"; $repo = "$HOME\english-wheel-closed-loop"
# Robot code: the pinned, verified export of Danish's driver by default (his working folder changes while he develops);
#   -RobotCode live  = his working folder; -RobotCode <path> = another export. The adapter refuses any version that is not
#   in robot_dobot/verified_robot_code.json unless -AllowUnverifiedRobotCode (only with Danish, after he says it is safe).
$scripts = if ($RobotCode -eq "live") { "$HOME\Documents\GitHub\English-Wheel---DoBot-Files\Scripts" } elseif ($RobotCode) { $RobotCode } else { "$HOME\ew_robot_code_pinned\bbe1380\Scripts" }
if ($AllowUnverifiedRobotCode) { $env:EW_ALLOW_UNVERIFIED_ROBOT_CODE = "1" } else { Remove-Item Env:EW_ALLOW_UNVERIFIED_ROBOT_CODE -ErrorAction SilentlyContinue }
"robot code: $scripts"
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
