# Run elevated. NVIDIA Compute Sanitizer requires this Windows debugger interface.
# https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html#windows-specific-behavior
$ErrorActionPreference = 'Stop'
$key = 'HKLM:\SOFTWARE\NVIDIA Corporation\GPUDebugger'
$log = Join-Path $PSScriptRoot 'results\debugger-interface-setup.json'
$old = (Get-ItemProperty -LiteralPath $key -Name EnableInterface -ErrorAction SilentlyContinue).EnableInterface
if (-not (Test-Path -LiteralPath $key)) { New-Item -Path $key -Force | Out-Null }
New-ItemProperty -LiteralPath $key -Name EnableInterface -PropertyType DWord -Value 1 -Force | Out-Null
$new = (Get-ItemProperty -LiteralPath $key -Name EnableInterface).EnableInterface
@{ previousValue=$old; newValue=$new; timestampUtc=[DateTime]::UtcNow.ToString('o'); registryPath=$key } | ConvertTo-Json | Set-Content -LiteralPath $log
if ($new -ne 1) { throw 'The debugger interface was not enabled.' }
