param(
    [ValidateSet('Upload','Run','Status','CheckLocal')][string]$Stage = 'Upload'
)
$ErrorActionPreference='Stop'
$Workspace='C:\Users\Tripozik\Desktop\MWS'
$Key=Join-Path $Workspace 'hackathon-235824-Tripozik.pem'
$Target='ubuntu@81.94.159.247'
$Bundle=Join-Path $PSScriptRoot '005-targeted-trainonly-bundle.tar.gz'
$Installer=Join-Path $PSScriptRoot 'install-targeted-005.py'
$Options=@('-i',$Key,'-o','BatchMode=no','-o','ConnectTimeout=15','-o','ServerAliveInterval=20')
$Python='/home/ubuntu/mws/.venv/bin/python'
$Runner='/home/ubuntu/mws-v21/targeted/005-targeted-001-trainonly/run_targeted.py'
$Run='/home/ubuntu/mws-v21/out/005-targeted-001'
function Invoke-Checked([string]$Program,[string[]]$Arguments) {
    & $Program @Arguments
    if($LASTEXITCODE -ne 0){throw "$Program failed (exit $LASTEXITCODE). Existing results preserved."}
}
if($Stage -eq 'Upload' -or $Stage -eq 'CheckLocal') {
    foreach($File in @($Bundle,($Bundle+'.sha256'),$Installer)) {
        if(!(Test-Path -LiteralPath $File -PathType Leaf)){throw "Missing file: $File"}
    }
    $Expected=(Get-Content -Raw -LiteralPath ($Bundle+'.sha256')).Trim()
    if($Expected -notmatch '^[0-9a-fA-F]{64}$'){throw 'Invalid checksum file'}
    if((Get-FileHash -Algorithm SHA256 -LiteralPath $Bundle).Hash -ne $Expected){throw 'Bundle SHA256 mismatch'}
    if($Stage -eq 'CheckLocal'){Write-Host 'LOCAL PACKAGE SHA256 OK. No connection or training.';exit 0}
    Invoke-Checked 'scp' ($Options+@($Bundle,($Target+':005-targeted-001-bundle.tar.gz')))
    Invoke-Checked 'scp' ($Options+@($Installer,($Target+':install-targeted-005.py')))
    Invoke-Checked 'ssh' ($Options+@($Target,"$Python /home/ubuntu/install-targeted-005.py --archive /home/ubuntu/005-targeted-001-bundle.tar.gz --sha256 $Expected"))
    Invoke-Checked 'ssh' ($Options+@($Target,"$Python $Runner --check-only"))
    Write-Host 'INSTALLED AND CHECKED. Training has NOT started.'
} elseif($Stage -eq 'Run') {
    Invoke-Checked 'ssh' ($Options+@($Target,"$Python $Runner --check-only"))
    $Job="umask 077; $Python -u $Runner --train-only >> $Run.log 2>&1"
    Invoke-Checked 'ssh' ($Options+@($Target,"tmux new-session -d -s 005-targeted-001 '$Job'"))
    Write-Host 'STARTED: token checks, 1-step smoke, full train, merge. NO VAL RUNS OR VAL LOSS.'
    Write-Host 'Existing merged parent is preserved. The candidate will not be promoted automatically.'
} else {
    Invoke-Checked 'ssh' ($Options+@($Target,"tail -n 25 $Run.log"))
}
