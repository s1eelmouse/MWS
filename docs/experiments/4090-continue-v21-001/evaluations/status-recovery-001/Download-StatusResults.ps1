param(
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9._-]*$')][string]$RunId='4090-v21-status-001',
    [string]$DownloadDir='C:\Users\Tripozik\Desktop\MWS\repository\docs\experiments\4090-continue-v21-001\evaluations\status-recovery-001\attachments\download',
    [string]$Key='C:\Users\Tripozik\Desktop\MWS\hackathon-235824-Tripozik.pem'
)
$ErrorActionPreference='Stop'
$Target='ubuntu@81.94.159.247'
$ArchiveName=$RunId+'-results.tar.gz'
$Archive=Join-Path $DownloadDir $ArchiveName
$Checksum=$Archive+'.sha256'
$Extracted=Join-Path $DownloadDir $RunId
$Options=@('-i',$Key,'-o','BatchMode=no','-o','ConnectTimeout=15','-o','ServerAliveInterval=20')
function Invoke-Checked([string]$Program,[string[]]$Arguments) {
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Program failed (exit $LASTEXITCODE). Preserve files and inspect the error." }
}
if (Test-Path -LiteralPath $Extracted) {
    throw "Results already exist: $Extracted. Use another -DownloadDir to preserve the previous download."
}
New-Item -ItemType Directory -Force -Path $DownloadDir | Out-Null
Invoke-Checked 'scp' ($Options+@(($Target+':'+$ArchiveName+'.sha256'),$Checksum))
$Expected=((Get-Content -Raw -LiteralPath $Checksum).Trim() -split '\s+')[0].ToLowerInvariant()
if ($Expected -notmatch '^[0-9a-f]{64}$') { throw 'Invalid server checksum file.' }
if (!(Test-Path -LiteralPath $Archive)) {
    $Partial=$Archive+'.partial'
    Invoke-Checked 'scp' ($Options+@(($Target+':'+$ArchiveName),$Partial))
    $Actual=(Get-FileHash -Algorithm SHA256 -LiteralPath $Partial).Hash.ToLowerInvariant()
    if ($Actual -ne $Expected) { throw 'Archive checksum mismatch. Preserve files; extraction was not started.' }
    Move-Item -LiteralPath $Partial -Destination $Archive
} else {
    $Actual=(Get-FileHash -Algorithm SHA256 -LiteralPath $Archive).Hash.ToLowerInvariant()
    if ($Actual -ne $Expected) { throw 'Existing archive differs from server checksum. Use another -DownloadDir.' }
}
$Entries=@(& tar -tzf $Archive)
if ($LASTEXITCODE -ne 0) { throw 'Could not inspect archive.' }
foreach ($Entry in $Entries) {
    $Parts=$Entry.Replace('\','/').Split('/')
    if ($Entry.StartsWith('/') -or $Entry.Contains(':') -or $Parts -contains '..' -or
        !($Entry -eq ($RunId+'.log') -or $Entry -eq $RunId -or $Entry.StartsWith($RunId+'/'))) {
        throw "Unexpected archive path: $Entry"
    }
}
Invoke-Checked 'tar' @('-xzf',$Archive,'-C',$DownloadDir)
foreach ($Relative in @('settings.json','release-manifest.json','stages.jsonl','pipeline-status.json',
    'tests.log','validation.log','validation/report.json','validation/metrics.json',
    'validation/predictions.jsonl','validation/raw-answers.jsonl','validation/calls.jsonl',
    'validation/batches.jsonl','validation/memory.csv')) {
    if (!(Test-Path -LiteralPath (Join-Path $Extracted $Relative))) { throw "Missing result file: $Relative" }
}
$Report=Get-Content -Raw -LiteralPath (Join-Path $Extracted 'validation/report.json') | ConvertFrom-Json
$Pipeline=Get-Content -Raw -LiteralPath (Join-Path $Extracted 'pipeline-status.json') | ConvertFrom-Json
if ($Pipeline.status -ne 'complete' -or $Report.status -ne 'complete') {
    throw 'Downloaded results are preserved, but the server report is not complete.'
}
$Inventory=@(Get-ChildItem -LiteralPath $Extracted -Recurse -File | ForEach-Object {
    [ordered]@{path=$_.FullName.Substring($Extracted.Length+1).Replace('\','/');bytes=$_.Length;
        sha256=(Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName).Hash.ToLowerInvariant()}
})
$Verification=[ordered]@{run_id=$RunId;downloaded_utc=[DateTime]::UtcNow.ToString('o');
    archive_sha256=$Actual;server_checksum_matches=$true;status=$Report.status;
    calls=$Report.quality.official.n_calls;failed=$Report.failed_calls;files=$Inventory}
$Verification | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 -LiteralPath (Join-Path $DownloadDir 'download-verification.json')
Write-Host 'DOWNLOADED, SHA256 VERIFIED AND EXTRACTED'
Write-Host "Results: $Extracted"
Write-Host "Calls: $($Verification.calls); failed: $($Verification.failed)"
Write-Host "Main report: $(Join-Path $Extracted 'validation/report.json')"
