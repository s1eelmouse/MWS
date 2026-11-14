param(
    [Parameter(Mandatory=$true)]
    [ValidateSet('PrepareOld','Download','Code','UploadModel','Install','ResumeInstall')]
    [string]$Stage,
    [ValidatePattern('^[A-Za-z0-9_-]+$')][string]$RunId = '001',
    [string]$DownloadDir = 'D:\MWS\team_shpana-transfer',
    [string]$OldKey = 'C:\Users\Tripozik\Desktop\MWS\hackathon-235824-Tripozik.pem',
    [string]$CodePackage = 'C:\Users\Tripozik\Desktop\MWS\outputs\team-shpana-code-001.tar.gz'
)
$ErrorActionPreference = 'Stop'
$OldHost = 'ubuntu@81.94.159.247'
$Options = @('-o','ConnectTimeout=15','-o','ServerAliveInterval=20')
$OldOptions = @('-o','BatchMode=yes') + $Options
$NewOptions = @('-o','BatchMode=no') + $Options
$RemoteArchive = "/home/ubuntu/team-shpana-transfer/model-002-$RunId.tar.gz"
$ArchiveName = "model-002-$RunId.tar.gz"
$ModelTool = Join-Path $PSScriptRoot 'model-transfer.py'
if (!(Test-Path -LiteralPath $ModelTool)) {
    $ModelTool = 'C:\Users\Tripozik\Desktop\MWS\repository\docs\migrations\4090-to-v100\model-transfer.py'
}
function Invoke-Checked([string]$Program, [string[]]$Arguments) {
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Program failed (exit $LASTEXITCODE). Keep files and inspect the error." }
}
switch ($Stage) {
    'PrepareOld' {
        Invoke-Checked 'ssh' (@('-i',$OldKey) + $OldOptions + @($OldHost,"umask 077`nmkdir -p /home/ubuntu/team-shpana-transfer"))
        Invoke-Checked 'scp' (@('-i',$OldKey) + $OldOptions + @($ModelTool,"${OldHost}:/home/ubuntu/team-shpana-transfer/model-transfer.py"))
        $job = "python3 /home/ubuntu/team-shpana-transfer/model-transfer.py pack --output $RemoteArchive >> /home/ubuntu/team-shpana-transfer/pack-$RunId.log 2>&1"
        Invoke-Checked 'ssh' (@('-i',$OldKey) + $OldOptions + @($OldHost,"tmux new-session -d -s team-shpana-pack-$RunId '$job'"))
        Write-Host "Packing started; only model/adapter/data. Old-server log: /home/ubuntu/team-shpana-transfer/pack-$RunId.log"
    }
    'Download' {
        if ([IO.Path]::GetPathRoot([IO.Path]::GetFullPath($DownloadDir)) -ne 'D:\') { throw 'Use a destination on drive D:.' }
        New-Item -ItemType Directory -Force -Path $DownloadDir | Out-Null
        $local = Join-Path $DownloadDir $ArchiveName
        $manifestPath = $local + '.manifest.json'
        Invoke-Checked 'scp' (@('-i',$OldKey) + $OldOptions + @("${OldHost}:$RemoteArchive.manifest.json",$manifestPath))
        $manifest = Get-Content -Raw -LiteralPath $manifestPath | ConvertFrom-Json
        if ($manifest.archive -ne $ArchiveName) { throw 'Unexpected archive name.' }
        if ((Test-Path -LiteralPath $local) -and (Get-FileHash -Algorithm SHA256 -LiteralPath $local).Hash.ToLowerInvariant() -eq $manifest.archive_sha256) {
            Write-Host "Already downloaded and verified: $local"
            break
        }
        if ((Get-PSDrive D).Free -lt ($manifest.archive_bytes + 268435456)) { throw 'Not enough free space on D:.' }
        $partial = $local + '.partial'
        Invoke-Checked 'scp' (@('-i',$OldKey) + $OldOptions + @("${OldHost}:$RemoteArchive",$partial))
        if ((Get-Item -LiteralPath $partial).Length -ne $manifest.archive_bytes -or
            (Get-FileHash -Algorithm SHA256 -LiteralPath $partial).Hash.ToLowerInvariant() -ne $manifest.archive_sha256) { throw 'Downloaded model SHA256 mismatch.' }
        if (Test-Path -LiteralPath $local) { Move-Item -LiteralPath $local -Destination ($local + '.previous-' + [DateTime]::UtcNow.Ticks) }
        Move-Item -LiteralPath $partial -Destination $local
        Write-Host "MODEL PACKAGE VERIFIED ON D: $local"
    }
    'Code' {
        if (!(Test-Path -LiteralPath $CodePackage)) { throw "Code package missing: $CodePackage" }
        Invoke-Checked 'scp' ($NewOptions + @($CodePackage,'inh-compute:team-shpana-code-001.tar.gz'))
        $remote = @'
set -eu
umask 077
if [ -e "$HOME/team_shpana/src" ]; then
    echo 'team_shpana/src already exists. Preserve it and review before replacing code.'
    exit 1
fi
mkdir -p "$HOME/team_shpana"
chmod 700 "$HOME/team_shpana"
tar -xzf "$HOME/team-shpana-code-001.tar.gz" -C "$HOME"
echo 'Code installed in ~/team_shpana. GPU jobs have not started.'
'@
        Invoke-Checked 'ssh' ($NewOptions + @('inh-compute',$remote))
    }
    'UploadModel' {
        $local = Join-Path $DownloadDir $ArchiveName
        $manifestPath = $local + '.manifest.json'
        $manifest = Get-Content -Raw -LiteralPath $manifestPath | ConvertFrom-Json
        if ((Get-FileHash -Algorithm SHA256 -LiteralPath $local).Hash.ToLowerInvariant() -ne $manifest.archive_sha256) { throw 'Local model checksum mismatch.' }
        Invoke-Checked 'ssh' ($NewOptions + @('inh-compute',"umask 077`nmkdir -p ~/team_shpana/incoming"))
        Invoke-Checked 'scp' ($NewOptions + @($local,"inh-compute:team_shpana/incoming/$ArchiveName"))
        Invoke-Checked 'scp' ($NewOptions + @($manifestPath,"inh-compute:team_shpana/incoming/$ArchiveName.manifest.json"))
        $remote = "python3 ~/team_shpana/docs/migrations/4090-to-v100/model-transfer.py install --archive ~/team_shpana/incoming/$ArchiveName --root ~/team_shpana"
        Invoke-Checked 'ssh' ($NewOptions + @('inh-compute',$remote))
    }
    'Install' {
        $remote = "tmux new-session -d -s team-shpana-install-$RunId 'umask 077; bash ~/team_shpana/docs/migrations/4090-to-v100/install-v100.sh >> ~/team_shpana/install-$RunId.log 2>&1'"
        Invoke-Checked 'ssh' ($NewOptions + @('inh-compute',$remote))
        Write-Host "Install started in tmux. New-server log: ~/team_shpana/install-$RunId.log"
    }
    'ResumeInstall' {
        $resume = Join-Path $PSScriptRoot 'resume-v100-install.sh'
        if (!(Test-Path -LiteralPath $resume)) {
            $resume = 'C:\Users\Tripozik\Desktop\MWS\repository\docs\migrations\4090-to-v100\resume-v100-install.sh'
        }
        Invoke-Checked 'scp' ($NewOptions + @($resume,'inh-compute:team_shpana/resume-v100-install.sh'))
        $remote = "tmux new-session -d -s team-shpana-resume-$RunId 'umask 077; bash ~/team_shpana/resume-v100-install.sh >> ~/team_shpana/install-resume-$RunId.log 2>&1'"
        Invoke-Checked 'ssh' ($NewOptions + @('inh-compute',$remote))
        Write-Host "Installation resumed in existing venv. Log: ~/team_shpana/install-resume-$RunId.log"
    }
}
