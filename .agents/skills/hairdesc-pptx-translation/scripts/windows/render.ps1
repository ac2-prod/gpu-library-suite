param(
    [Parameter(Mandatory=$true)][string]$Source,
    [Parameter(Mandatory=$true)][string]$Output,
    [string]$Python = 'python',
    [switch]$Refit
)
$ErrorActionPreference = 'Stop'
Import-Module (Join-Path $PSScriptRoot 'Renderer.psm1')
$session = Enter-RenderSession
try { Invoke-OwnedRender -Source $Source -Output $Output -Python $Python -Refit:$Refit }
finally { $session.ReleaseMutex(); $session.Dispose() }
