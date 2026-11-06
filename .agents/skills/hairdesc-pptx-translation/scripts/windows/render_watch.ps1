param(
    [Parameter(Mandatory=$true)][string]$Root,
    [string]$Python='python',
    [ValidateRange(1,3600)][int]$IntervalSec=5,
    [ValidateRange(1,3600)][int]$SettleSec=3
)
$ErrorActionPreference='Stop'
Import-Module (Join-Path $PSScriptRoot 'Renderer.psm1')
$rootItem=Get-Item -LiteralPath $Root
if (-not $rootItem.PSIsContainer -or $rootItem.FullName.StartsWith('\\') -or
    ([IO.DriveInfo]::new([IO.Path]::GetPathRoot($rootItem.FullName))).DriveType -notin @('Fixed','Removable','Ram') -or
    ($rootItem.Attributes -band [IO.FileAttributes]::ReparsePoint)) { throw 'Root must be a local QA directory' }
function Get-Checks([string]$Directory) {
    foreach ($item in (Get-ChildItem -LiteralPath $Directory)) {
        if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) { continue }
        if ($item.PSIsContainer) { Get-Checks $item.FullName }
        elseif ($item.Name -ieq 'render_check.pptx') { $item }
    }
}
$session=Enter-RenderSession
try {
    while ($true) {
        foreach ($src in (Get-Checks $rootItem.FullName)) {
            $pdf=[IO.Path]::ChangeExtension($src.FullName,'.pdf')
            $age=([DateTime]::UtcNow-$src.LastWriteTimeUtc).TotalSeconds
            if ($age -lt $SettleSec) { continue }
            if ((Test-Path -LiteralPath $pdf) -and (Get-Item -LiteralPath $pdf).LastWriteTimeUtc -ge $src.LastWriteTimeUtc) { continue }
            try {
                Invoke-OwnedRender -Source $src.FullName -Output $pdf -Python $Python
                Add-Content -LiteralPath (Join-Path $src.DirectoryName 'render_check.log') -Value ("{0:o} OK" -f [DateTime]::UtcNow)
            } catch {
                Add-Content -LiteralPath (Join-Path $src.DirectoryName 'render_check.log') -Value ("{0:o} STOP {1}" -f [DateTime]::UtcNow,$_.Exception.Message)
                throw
            }
        }
        Start-Sleep -Seconds $IntervalSec
    }
} finally { $session.ReleaseMutex(); $session.Dispose() }
