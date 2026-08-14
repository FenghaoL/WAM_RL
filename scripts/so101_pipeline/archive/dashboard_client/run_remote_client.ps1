$ErrorActionPreference = "Stop"

# Edit defaults here; this script intentionally takes no command arguments.
$Task = "Pick up the white block."
# Other cached task: "Put the white block on the paper roll."
$Mode = "action"  # action | video_preview | infer_with_video
$PolicyPort = 12000  # action service: 12000 | video diagnostic service: 12001
$EnableMotion = $true

$FollowerPort = "COM4"
$FollowerId = "fenghao_so101_follower"
$FixedCameraIndex = 2
$WristCameraIndex = 0
$FixedCropRightPx = 160
$ImageTransport = "front"  # front is the exact 384x320 training layout; raw lets the server compose it.
$CameraFps = 30
$FixedAutoExposure = 0.25
$WristAutoExposure = 0.25
$FixedExposure = -6.0
$WristExposure = -5.0

$ActionFps = 15
$ExpectedChunkSize = 8
$ExpectedActionHorizon = 40
$MaxStepDeg = 3.0
$JointLimitMargin = 0.0
$HomeOnStartup = $false
$HomeArm = @(3.1, -34.3, 31.5, 55.9, -12.3, 13.4)
$HomeIgnoreGripper = $true
$HomeTimeoutS = 15
$HomeTolerance = 2.0
$HomeStep = 5.0
$HomeStepIntervalS = 0.02
$MaxChunks = 0  # 0 runs until Stop is pressed.
$ActionTimeoutS = 120
$VideoTimeoutS = 1800
$DashboardFps = 10
$DashboardCameraFps = 8
$DashboardHistory = 50
$PrintServerResponses = $true
$PreviewOutputDir = "outputs\ahawam_previews"
$TimingLog = "outputs\ahawam_client_timing.jsonl"

$EnvironmentName = "so101-ahawam-collect"
$EnvironmentPath = Join-Path $env:USERPROFILE ".conda\envs\$EnvironmentName"
$Python = Join-Path $EnvironmentPath "python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
  throw "Missing $EnvironmentName at $EnvironmentPath. Install the local LeRobot SO101 environment first."
}
$EnvironmentBins = @(
  $EnvironmentPath,
  (Join-Path $EnvironmentPath "Library\bin"),
  (Join-Path $EnvironmentPath "Scripts")
)
$env:PATH = ($EnvironmentBins + @($env:PATH)) -join ";"

$Client = Join-Path $PSScriptRoot "ahawam_dashboard_client.py"
$PreviewOutputDir = Join-Path $PSScriptRoot $PreviewOutputDir
$TimingLog = Join-Path $PSScriptRoot $TimingLog

if (($Mode -eq "video_preview" -or $Mode -eq "infer_with_video") -and $PolicyPort -ne 12001) {
  throw "Video diagnostic modes require PolicyPort 12001."
}
if ($EnableMotion -and $PolicyPort -ne 12000) {
  throw "Live action is restricted to the action service on PolicyPort 12000."
}

$TimeoutS = $ActionTimeoutS
if ($Mode -eq "video_preview" -or $Mode -eq "infer_with_video") {
  $TimeoutS = $VideoTimeoutS
}

$ClientArgs = @(
  $Client,
  "--host", "127.0.0.1",
  "--port", "$PolicyPort",
  "--task", $Task,
  "--mode", $Mode,
  "--robot-port", $FollowerPort,
  "--robot-id", $FollowerId,
  "--fixed-camera", "$FixedCameraIndex",
  "--wrist-camera", "$WristCameraIndex",
  "--fixed-crop-right-px", "$FixedCropRightPx",
  "--image-transport", $ImageTransport,
  "--camera-fps", "$CameraFps",
  "--fixed-auto-exposure", "$FixedAutoExposure",
  "--wrist-auto-exposure", "$WristAutoExposure",
  "--fixed-exposure", "$FixedExposure",
  "--wrist-exposure", "$WristExposure",
  "--action-fps", "$ActionFps",
  "--expected-chunk-size", "$ExpectedChunkSize",
  "--expected-action-horizon", "$ExpectedActionHorizon",
  "--max-step", "$MaxStepDeg",
  "--joint-limit-margin", "$JointLimitMargin",
  "--home-timeout-s", "$HomeTimeoutS",
  "--home-tolerance", "$HomeTolerance",
  "--home-step", "$HomeStep",
  "--home-step-interval-s", "$HomeStepIntervalS",
  "--max-chunks", "$MaxChunks",
  "--timeout-s", "$TimeoutS",
  "--dashboard-fps", "$DashboardFps",
  "--dashboard-camera-fps", "$DashboardCameraFps",
  "--dashboard-history", "$DashboardHistory",
  "--preview-output-dir", $PreviewOutputDir,
  "--timing-log", $TimingLog
)

$ClientArgs += "--home-arm"
foreach ($value in $HomeArm) {
  $ClientArgs += "$value"
}
if ($HomeOnStartup) {
  $ClientArgs += "--home-on-startup"
}
if ($HomeIgnoreGripper) {
  $ClientArgs += "--home-ignore-gripper"
}

if ($PrintServerResponses) {
  $ClientArgs += "--print-server-responses"
}

if ($EnableMotion) {
  Write-Warning "LIVE MOTION ENABLED. The client will execute complete 8-step chunks at 15 Hz."
  $null = Read-Host "Workspace clear? Confirm the camera layout and emergency stop, then press Enter"
} else {
  $ClientArgs += "--dry-run"
  Write-Warning "DRY RUN: actions will be shown but never sent to the follower."
}

Write-Host "AHA-WAM SO-101 remote client"
Write-Host "  policy:      127.0.0.1:$PolicyPort (SSH tunnel endpoint)"
Write-Host "  task:        $Task"
Write-Host "  mode:        $Mode"
Write-Host "  robot:       $FollowerPort"
Write-Host "  cameras:     fixed=$FixedCameraIndex crop-right=$FixedCropRightPx; wrist=$WristCameraIndex"
Write-Host "  transport:   $ImageTransport"
Write-Host "  frame:       SO101 calibrated degrees (identity mapping)"
Write-Host "  model view:  fixed 320x256; wrist 160x128 left; black 160x128 right"
Write-Host "  timing:      $ActionFps Hz; server chunk=$ExpectedChunkSize; horizon=$ExpectedActionHorizon"
Write-Host "  control:     max tick delta=$MaxStepDeg; calibration limits are display-only"
Write-Host "  motion:      $EnableMotion"
Write-Host "  home:        optional; startup=$($EnableMotion -and $HomeOnStartup); pose=$($HomeArm -join ', ')"

& $Python @ClientArgs
exit $LASTEXITCODE
