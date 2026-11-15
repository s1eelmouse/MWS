param([Parameter(Mandatory=$true)][ValidateSet('Upload','Run')][string]$Stage)
$ErrorActionPreference='Stop'
$Target='ubuntu@81.94.159.247'
$Key='C:\Users\Tripozik\Desktop\MWS\hackathon-235824-Tripozik.pem'
$Options=@('-i',$Key,'-o','BatchMode=no','-o','ConnectTimeout=15','-o','ServerAliveInterval=20')
function Invoke-Checked([string]$Program,[string[]]$Arguments) {
 & $Program @Arguments
 if($LASTEXITCODE -ne 0){throw "$Program failed (exit $LASTEXITCODE). Preserve outputs and inspect the error."}
}
$Remote='/home/ubuntu/mws-v21/run-005-plain-epoch-001.py'
if($Stage -eq 'Upload') {
 Invoke-Checked 'scp' ($Options+@((Join-Path $PSScriptRoot 'run-005-plain-epoch-001.py'),($Target+':mws-v21/run-005-plain-epoch-001.py')))
 Invoke-Checked 'scp' ($Options+@((Join-Path $PSScriptRoot 'train-v21-original.jsonl.gz'),($Target+':mws-v21/train-v21-original.jsonl.gz')))
 Invoke-Checked 'ssh' ($Options+@($Target,"/home/ubuntu/mws/.venv/bin/python $Remote --install-train"))
 Write-Host 'ORIGINAL TRAIN AND RUNNER INSTALLED. Training has not started.'
} else {
 Invoke-Checked 'ssh' ($Options+@($Target,"/home/ubuntu/mws/.venv/bin/python $Remote --check-only"))
 $Job="umask 077; /home/ubuntu/mws/.venv/bin/python -u $Remote >> /home/ubuntu/mws-v21/out/005-plain-epoch-001.log 2>&1"
 Invoke-Checked 'ssh' ($Options+@($Target,"tmux new-session -d -s 005-plain-epoch-001 '$Job'"))
 Write-Host 'STARTED: 1 plain epoch from merged 005, merge, one full clean validation.'
 Write-Host 'Log: tail -f ~/mws-v21/out/005-plain-epoch-001.log'
}
