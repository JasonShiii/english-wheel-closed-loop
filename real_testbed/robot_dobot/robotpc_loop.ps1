param([string]$OutName = "sheet01", [string]$Target = "S1_t042_c4", [int]$N = 3, [int]$MaxPasses = 12,
      [switch]$Resume, [switch]$Confirm, [switch]$AskRadii, [switch]$Show)
# The closed loop on the Dobot cell. From the DESKTOP terminal the force monitors and (with -Show) his preview are visible:
#   powershell -ExecutionPolicy Bypass -File C:\Users\"Ping Guo Lab"\robotpc_loop.ps1 -OutName sheet01 -Confirm -Show
# -Confirm: the loop asks before every block; -AskRadii: his Rx/Rz prompt between passes; log = loops\<OutName>.log
$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"; $env:PYTHONUTF8 = "1"; [Console]::OutputEncoding = [Text.Encoding]::UTF8
$py = "C:\anaconda3\envs\EnglishWheel\python.exe"; $repo = "$HOME\english-wheel-closed-loop"; $scripts = "$HOME\Documents\GitHub\English-Wheel---DoBot-Files\Scripts"
Set-Location "$repo\real_testbed"
$out = "$repo\loops\$OutName"
$extra = @(); if ($Resume) { $extra += "--resume" }; if (-not $Confirm) { $extra += "--yes" }; if (-not $AskRadii) { $extra += "--auto-radii" }; if ($Show) { $extra += "--show-preview" }
$t0 = Get-Date
& $py -m ew_loop.cli rigloop --plant dobot --robot-scripts $scripts --server http://localhost:8765 `
    --target "figures/rig/targets/$Target.npz" --out $out --calib "pc_post_processing/deliverable_092826/calibration results" `
    --clamp 12 --n $N --max-passes $MaxPasses @extra 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath "$out.log" -Append |
    Where-Object { $_ -notmatch "UserWarning|plt.show|fig, \(ax_f|generic_filter|_nd_image|RuntimeWarning|^\s*$" }
"exit $LASTEXITCODE after {0:N0} s" -f ((Get-Date) - $t0).TotalSeconds
