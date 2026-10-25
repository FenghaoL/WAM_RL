$ErrorActionPreference = "Stop"

# Options: annotate, audit, export. Edit defaults here; no arguments required.
$Mode = "annotate"
$Labels = @("success", "insert", "release", "recovery")
$DatasetBase = "so101_data\ahawam_hil_raw\so101_ahawam_cylinder_hil_v1"
$Runs = @(
  "20260911_183057"
  "20260911_225809"
  "20260911_232927"
  "20260912_001855"
  "20260912_010503"
  "20260912_010747"
)

$RepoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\..\.."))
$EnvironmentPath = Join-Path $env:USERPROFILE ".conda\envs\so101-ahawam-collect"
$Python = Join-Path $EnvironmentPath "python.exe"
if (-not (Test-Path -LiteralPath $Python)) { throw "Missing Python: $Python" }
$env:PATH = (@($EnvironmentPath, (Join-Path $EnvironmentPath "Library\bin"), (Join-Path $EnvironmentPath "Scripts"), $env:PATH) -join ";")
$ClientArgs = @((Join-Path $PSScriptRoot "annotate.py"))
foreach ($run in $Runs) {
  $dataset = Join-Path (Join-Path $RepoRoot $DatasetBase) $run
  $ClientArgs += @("--dataset", $dataset)
}
if ($Mode -eq "export") { $ClientArgs += "--export" }
elseif ($Mode -eq "audit") { $ClientArgs += "--audit" }
elseif ($Mode -ne "annotate") { throw "Unknown mode: $Mode" }
$ClientArgs += "--labels"
$ClientArgs += $Labels
& $Python @ClientArgs
if ($LASTEXITCODE -ne 0) { throw "Annotation tool exited with code $LASTEXITCODE" }
