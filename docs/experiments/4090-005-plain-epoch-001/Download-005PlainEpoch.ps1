param(
 [string]$BulkDir='C:\Users\Tripozik\Desktop\MWS\server-results\005-plain-epoch-001',
 [string]$ReportDir='C:\Users\Tripozik\Desktop\MWS\repository\docs\experiments\4090-005-plain-epoch-001\attachments\download\005-plain-epoch-001',
 [string]$Key='C:\Users\Tripozik\Desktop\MWS\hackathon-235824-Tripozik.pem'
)
$ErrorActionPreference='Stop'
$RunId='005-plain-epoch-001'
$Target='ubuntu@81.94.159.247'
$Options=@('-i',$Key,'-o','BatchMode=no','-o','ConnectTimeout=15','-o','ServerAliveInterval=20')
$Python='C:\Program Files\WindowsApps\OpenAI.CodexPrimaryRuntime.v26-1112-737-0_26.1112.737.0_x64__3k8sg7r9htsxt\dependencies\python\python.exe'
function Invoke-Checked([string]$Program,[string[]]$Arguments) {
 & $Program @Arguments
 if($LASTEXITCODE -ne 0){throw "$Program failed (exit $LASTEXITCODE). Preserve files and inspect the error."}
}
if(!(Test-Path -LiteralPath $Python)){throw 'Local extraction Python is unavailable; no files were changed.'}
$Archive=Join-Path $BulkDir ($RunId+'-results.tar.gz')
$Extracted=Join-Path $BulkDir 'extracted'
$Full=Join-Path $Extracted $RunId
if(Test-Path -LiteralPath $ReportDir){throw "Report directory already exists; preserve it: $ReportDir"}
New-Item -ItemType Directory -Force -Path $BulkDir | Out-Null
$RemoteArchive="/home/ubuntu/$RunId-results.tar.gz"
$Packer='/home/ubuntu/mws-v21/releases/case-v21-unified-001/evaluation/pack_results.py'
# This frozen packer includes every regular file inside out/RunId, including
# adapter checkpoints and merged weights for this training experiment.
Invoke-Checked 'ssh' ($Options+@($Target,"test -s $RemoteArchive.sha256 || /home/ubuntu/mws/.venv/bin/python $Packer --run-id $RunId"))
Invoke-Checked 'scp' ($Options+@(($Target+':'+$RemoteArchive+'.sha256'),($Archive+'.sha256')))
$Expected=((Get-Content -Raw -LiteralPath ($Archive+'.sha256')).Trim() -split '\s+')[0].ToLowerInvariant()
if($Expected -notmatch '^[0-9a-f]{64}$'){throw 'Invalid archive checksum.'}
if(!(Test-Path -LiteralPath $Archive)) {
 $Partial=$Archive+'.partial'
 Invoke-Checked 'scp' ($Options+@(($Target+':'+$RemoteArchive),$Partial))
 if((Get-FileHash -Algorithm SHA256 -LiteralPath $Partial).Hash.ToLowerInvariant() -ne $Expected){throw 'Archive checksum mismatch; partial preserved.'}
 Move-Item -LiteralPath $Partial -Destination $Archive
}
if((Get-FileHash -Algorithm SHA256 -LiteralPath $Archive).Hash.ToLowerInvariant() -ne $Expected){throw 'Existing archive checksum mismatch; preserve it and use another BulkDir.'}
if(!(Test-Path -LiteralPath $Extracted)) {
 Invoke-Checked $Python @('-B','-X','utf8',(Join-Path $PSScriptRoot 'Extract-005Results.py'),
  '--archive',$Archive,'--sha256',$Expected,'--destination',$Extracted)
} else {
 $VerificationPath=Join-Path $BulkDir 'download-verification.json'
 if(!(Test-Path -LiteralPath $VerificationPath)){throw 'Existing extraction has no completion manifest; preserve it and use another BulkDir.'}
 $Verification=Get-Content -Raw -LiteralPath $VerificationPath | ConvertFrom-Json
 if($Verification.archive_sha256 -ne $Expected){throw 'Existing extraction belongs to another archive.'}
 foreach($File in $Verification.files) {
  if((Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $Extracted $File.path)).Hash.ToLowerInvariant() -ne $File.sha256){throw "Extracted file changed: $($File.path)"}
 }
}
foreach($File in @('settings.json','release-manifest.json','stages.jsonl','pipeline-status.json',
 'comparison.json','tests.log','data-check.log','train.log','merge.log','validation.log',
 'adapter/adapter_model.safetensors','adapter/adapter_config.json','adapter/experiment.json',
 'merged/model.safetensors','merged/config.json','merged/tokenizer.json',
 'validation/report.json','validation/metrics.json','validation/raw-answers.jsonl',
 'validation/predictions.jsonl','validation/predictions-before-series.jsonl',
 'validation/calls.jsonl','validation/batches.jsonl','validation/memory.csv','validation/environment.json')) {
 if(!(Test-Path -LiteralPath (Join-Path $Full $File))){throw "Missing result: $File. Full download is preserved."}
}
$Status=Get-Content -Raw -LiteralPath (Join-Path $Full 'pipeline-status.json') | ConvertFrom-Json
$Report=Get-Content -Raw -LiteralPath (Join-Path $Full 'validation/report.json') | ConvertFrom-Json
if($Status.status -ne 'complete' -or $Report.status -ne 'complete' -or $Report.quality.official.n_calls -ne 383){throw 'Full results preserved, but the full cycle did not complete.'}
if(@(Get-ChildItem -LiteralPath (Join-Path $Full 'adapter') -Directory -Filter 'attempt-*').Count -eq 0){throw 'Training measurement directory is missing.'}
$Parent=Split-Path -Parent $ReportDir
New-Item -ItemType Directory -Force -Path $Parent | Out-Null
$Ignore=Join-Path $Parent '.gitignore'
if(!(Test-Path -LiteralPath $Ignore)){[IO.File]::WriteAllText($Ignore,"*`n!.gitignore`n",[Text.UTF8Encoding]::new($false))}
New-Item -ItemType Directory -Path $ReportDir | Out-Null
foreach($File in Get-ChildItem -LiteralPath $Full -Recurse -File) {
 $Relative=$File.FullName.Substring($Full.Length+1)
 if($File.Extension.ToLowerInvariant() -in @('.safetensors','.bin','.pt','.pth','.model')){continue}
 $Destination=Join-Path $ReportDir $Relative
 New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Destination) | Out-Null
 Copy-Item -LiteralPath $File.FullName -Destination $Destination
}
if(Test-Path -LiteralPath (Join-Path $Extracted ($RunId+'.log'))) {
 Copy-Item -LiteralPath (Join-Path $Extracted ($RunId+'.log')) -Destination (Join-Path $ReportDir 'pipeline.log')
}
Copy-Item -LiteralPath (Join-Path $BulkDir 'download-verification.json') -Destination (Join-Path $ReportDir 'download-verification.json')
[ordered]@{run_id=$RunId;archive=$Archive;archive_sha256=$Expected;complete_results=$Full;
 merged_model=(Join-Path $Full 'merged');adapter=(Join-Path $Full 'adapter');
 checkpoints=(Join-Path $Full 'adapter/checkpoints');report_copy=$ReportDir;
 weight_files_in_bulk_directory=$true;calls=$Report.quality.official.n_calls;failed_calls=$Report.failed_calls} |
 ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 -LiteralPath (Join-Path $ReportDir 'LOCAL_ARTIFACTS.json')
Write-Host 'DOWNLOAD COMPLETE: SHA256 verified, reports and new model weights preserved.'
Write-Host "FULL RESULTS AND WEIGHTS: $Full"
Write-Host "REPORT COPY: $ReportDir"
Write-Host "Calls: $($Report.quality.official.n_calls); failed: $($Report.failed_calls)"
Get-Content -Raw -LiteralPath (Join-Path $ReportDir 'comparison.json')
