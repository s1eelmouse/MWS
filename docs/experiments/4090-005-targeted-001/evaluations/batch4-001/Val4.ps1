param([ValidateSet('Upload','Noisy','Clean','Status')][string]$Stage='Upload')
$ErrorActionPreference='Stop'
$Target='ubuntu@81.94.159.247'
$Key='C:\Users\Tripozik\Desktop\MWS\hackathon-235824-Tripozik.pem'
$Options=@('-i',$Key,'-o','BatchMode=no','-o','ConnectTimeout=15','-o','ServerAliveInterval=20')
$Script=Join-Path $PSScriptRoot 'eval-targeted-b4.py'
$Remote='/home/ubuntu/mws-v21/eval-targeted-b4.py'
$Python='/home/ubuntu/mws/.venv/bin/python'
function Invoke-Checked([string]$Program,[string[]]$Arguments) {
 & $Program @Arguments
 if($LASTEXITCODE -ne 0){throw "$Program failed (exit $LASTEXITCODE). Existing results preserved."}
}
if($Stage -eq 'Upload') {
 if(!(Test-Path -LiteralPath $Script -PathType Leaf)){throw 'Evaluation script missing'}
 Invoke-Checked 'scp' ($Options+@($Script,($Target+':mws-v21/eval-targeted-b4.py')))
 Write-Host 'EVALUATION SCRIPT UPLOADED. No training files changed. No validation started.'
} elseif($Stage -eq 'Status') {
 Invoke-Checked 'ssh' ($Options+@($Target,'tail -n 25 /home/ubuntu/mws-v21/out/005-targeted-001-val-noisy-b4-001.log'))
} else {
 $Dataset=$Stage.ToLowerInvariant()
 $Session='targeted-val-'+$Dataset+'-b4'
 $Log='/home/ubuntu/mws-v21/out/005-targeted-001-val-'+$Dataset+'-b4-001.log'
 $Job="umask 077; $Python -u $Remote --dataset $Dataset >> $Log 2>&1"
 Invoke-Checked 'ssh' ($Options+@($Target,"tmux new-session -d -s $Session '$Job'"))
 Write-Host "QUEUED: $Dataset val at batch 4. Waits for training, probes memory, then runs ONE full val."
 Write-Host "Log: tail -n 40 -f $Log"
}
