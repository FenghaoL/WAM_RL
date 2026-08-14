$ErrorActionPreference = "Stop"

# Edit these values for the GPU server.
$Server = "chickadee.engin.umich.edu"
$User = "fenghao"
$ServiceMode = "action"  # action | video

if ($ServiceMode -eq "action") {
  $LocalPort = 12000
  $RemotePort = 12000
} elseif ($ServiceMode -eq "video") {
  $LocalPort = 12001
  $RemotePort = 12001
} else {
  throw "ServiceMode must be action or video."
}

$Forward = "127.0.0.1:${LocalPort}:127.0.0.1:${RemotePort}"
Write-Host "Forwarding local 127.0.0.1:$LocalPort to ${User}@${Server}:127.0.0.1:$RemotePort"
Write-Host "Keep this window open. Ctrl+C closes the tunnel."

& ssh -N `
  -o ExitOnForwardFailure=yes `
  -o ServerAliveInterval=30 `
  -o ServerAliveCountMax=3 `
  -L $Forward `
  "${User}@${Server}"

exit $LASTEXITCODE
