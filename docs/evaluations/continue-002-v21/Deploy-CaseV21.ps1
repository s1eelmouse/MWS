param(
    [Parameter(Mandatory=$true)][ValidateSet('Upload','Run')][string]$Stage,
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9._-]*$')][string]$RunId = 'continue-002-v21-full-val-002',
    [ValidateRange(1,30)][int]$BudgetGiB = 12,
    [ValidateRange(1,8)][int]$BatchSize = 1,
    [string]$Package = 'C:\Users\Tripozik\Desktop\MWS\outputs\team-shpana-case-v21-002.tar.gz'
)
$ErrorActionPreference = 'Stop'
$Options = @('-o','BatchMode=no','-o','ConnectTimeout=15','-o','ServerAliveInterval=20')
function Invoke-Checked([string]$Program,[string[]]$Arguments) {
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Program failed with exit $LASTEXITCODE. Preserve files and inspect the error." }
}
if ($Stage -eq 'Upload') {
    $expected = (Get-Content -Raw -LiteralPath ($Package + '.sha256')).Trim()
    if ($expected -notmatch '^[0-9a-f]{64}$' -or (Get-FileHash -Algorithm SHA256 -LiteralPath $Package).Hash.ToLowerInvariant() -ne $expected) {
        throw 'Local release package checksum mismatch.'
    }
    $installer = Join-Path $PSScriptRoot 'install-case-v21.py'
    if (!(Test-Path -LiteralPath $installer)) { $installer = 'C:\Users\Tripozik\Desktop\MWS\repository\docs\evaluations\continue-002-v21\install_release.py' }
    Invoke-Checked 'scp' ($Options + @($Package,'inh-compute:team_shpana/team-shpana-case-v21-002.tar.gz'))
    Invoke-Checked 'scp' ($Options + @($installer,'inh-compute:team_shpana/install-case-v21.py'))
    Invoke-Checked 'ssh' ($Options + @('inh-compute',"python3 ~/team_shpana/install-case-v21.py --archive ~/team_shpana/team-shpana-case-v21-002.tar.gz --expected-sha256 $expected"))
} else {
    $program = '$HOME/team_shpana/.venv-v100/bin/python'
    $runner = '$HOME/team_shpana/releases/case-v21-002/docs/evaluations/continue-002-v21/run_full_val.py'
    $log = '$HOME/team_shpana/out/' + $RunId + '.log'
    $job = "umask 077; $program -u $runner --run-id $RunId --budget-gib $BudgetGiB --batch-size $BatchSize >> $log 2>&1"
    Invoke-Checked 'ssh' ($Options + @('inh-compute',"tmux new-session -d -s $RunId '$job'"))
    Write-Host "EVALUATION ONLY started in tmux. Log: ~/team_shpana/out/$RunId.log"
}
