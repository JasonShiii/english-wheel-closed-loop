param([Parameter(Mandatory = $true)][string]$Commit)
# Export one commit of Danish's driver (read-only: git archive, his working folder is not touched) to
# ~/ew_robot_code_pinned/<commit>/Scripts and print the file hashes for robot_dobot/verified_robot_code.json.
# A new version is added there only after a rehearsal (dry run without a sheet) with Danish.
$repo = "$HOME\Documents\GitHub\English-Wheel---DoBot-Files"; $dst = "$HOME\ew_robot_code_pinned\$Commit"
if (Test-Path $dst) { "already pinned: $dst" } else {
    New-Item -ItemType Directory -Force -Path $dst | Out-Null
    $zip = "$env:TEMP\ew_$Commit.zip"
    git -C $repo archive --format=zip -o $zip $Commit Scripts
    Expand-Archive -Path $zip -DestinationPath $dst; Remove-Item $zip
    "pinned: $dst\Scripts"
}
& C:\anaconda3\envs\EnglishWheel\python.exe -c "import sys, json; sys.path.insert(0, r'$HOME\english-wheel-closed-loop\real_testbed\robot_dobot'); import run_toolpath_dobot as R, hashlib, os; ref = json.load(open(R.VERIFIED_CODE)); files = list(next(v for k, v in ref.items() if not k.startswith('_'))['files']); d = r'$dst\Scripts'; print(json.dumps({f: hashlib.sha256(open(os.path.join(d, f), 'rb').read().replace(b'\r\n', b'\n')).hexdigest() for f in files}, indent=1))"
