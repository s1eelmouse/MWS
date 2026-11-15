param(
    [Parameter(Mandatory=$true)][ValidateSet('Upload','Run')][string]$Stage,
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9._-]*$')][string]$RunId='4090-v21-status-001',
    [ValidateRange(1,8)][int]$BatchSize=2,
    [string]$Package='C:\Users\Tripozik\Desktop\MWS\outputs\team-shpana-case-v21-status-001.tar.gz',
    [string]$Key='C:\Users\Tripozik\Desktop\MWS\hackathon-235824-Tripozik.pem'
)
$ErrorActionPreference='Stop'
$Target='ubuntu@81.94.159.247'
$Options=@('-i',$Key,'-o','BatchMode=no','-o','ConnectTimeout=15','-o','ServerAliveInterval=20')
function Invoke-Checked([string]$Program,[string[]]$Arguments) {
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Program failed (exit $LASTEXITCODE). Preserve files and inspect the error." }
}
if ($Stage -eq 'Upload') {
    $Expected=(Get-Content -Raw -LiteralPath ($Package+'.sha256')).Trim()
    if ($Expected -notmatch '^[0-9a-f]{64}$' -or
        (Get-FileHash -Algorithm SHA256 -LiteralPath $Package).Hash.ToLowerInvariant() -ne $Expected) {
        throw 'Local package checksum mismatch.'
    }
    $Installer=Join-Path $PSScriptRoot 'install-status-release.py'
    if (!(Test-Path -LiteralPath $Installer)) { $Installer=Join-Path $PSScriptRoot 'install_release.py' }
    Invoke-Checked 'scp' ($Options+@($Package,($Target+':mws-v21/team-shpana-case-v21-status-001.tar.gz')))
    Invoke-Checked 'scp' ($Options+@($Installer,($Target+':mws-v21/install-status-release.py')))
    Invoke-Checked 'ssh' ($Options+@($Target,"python3 ~/mws-v21/install-status-release.py --archive ~/mws-v21/team-shpana-case-v21-status-001.tar.gz --expected-sha256 $Expected --team-root ~/mws-v21"))
} else {
    $Runner='/home/ubuntu/mws-v21/releases/case-v21-status-001/docs/experiments/4090-continue-v21-001/evaluations/status-recovery-001/run_full_val.py'
    $RemoteLog="/home/ubuntu/mws-v21/out/$RunId.log"
    $Job="umask 077; /home/ubuntu/mws/.venv/bin/python -u $Runner --run-id $RunId --batch-size $BatchSize >> $RemoteLog 2>&1"
    Invoke-Checked 'ssh' ($Options+@($Target,"tmux new-session -d -s $RunId '$Job'"))
    Write-Host "EVALUATION ONLY started. Log: $RemoteLog"
    Write-Host "Connect and read: tail -f $RemoteLog"
}
