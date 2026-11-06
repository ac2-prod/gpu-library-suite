# Private implementation shared by the direct command and watcher.
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
function Enter-RenderSession {
    $sid = [System.Security.Principal.WindowsIdentity]::GetCurrent().User.Value
    $mutex = [System.Threading.Mutex]::new($false, "Global\hairdesc-powerpoint-render-$sid")
    try {
        if (-not $mutex.WaitOne(0)) { throw 'PowerPoint renderer or watcher is already running' }
    } catch [System.Threading.AbandonedMutexException] {
        $mutex.ReleaseMutex(); $mutex.Dispose()
        throw 'Previous render session was interrupted; inspect its local stage before resuming'
    } catch { $mutex.Dispose(); throw }
    return $mutex
}
function Invoke-Guard([string]$Python, [string[]]$Arguments) {
    $guard = Join-Path $PSScriptRoot '..\common\render_guard.py'
    $value = & $Python -B $guard @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Render guard stopped ($LASTEXITCODE)" }
    return $value
}
function Get-Frames($shapes, $slideIdx, $prefix) {
    foreach ($sh in $shapes) {
        $sid = $prefix + [string]$sh.Id
        if ($sh.Type -eq 6) { Get-Frames $sh.GroupItems $slideIdx ($sid + '/'); continue }
        if ($sh.HasTable -eq -1) {
            $t = $sh.Table
            for ($r = 1; $r -le $t.Rows.Count; $r++) { for ($c = 1; $c -le $t.Columns.Count; $c++) {
                $cs = $t.Cell($r, $c).Shape
                [pscustomobject]@{ slide = $slideIdx; id = ('{0}[{1},{2}]' -f $sid, ($r - 1), ($c - 1)); shape = $cs; table = $true }
            } }
            continue
        }
        if ($sh.HasTextFrame -eq -1) { [pscustomobject]@{ slide = $slideIdx; id = $sid; shape = $sh; table = $false } }
    }
}

function Invoke-OwnedRender([string]$Source, [string]$Output, [string]$Python, [switch]$Refit) {
    $Source = (Resolve-Path -LiteralPath $Source).Path
    $Output = [IO.Path]::GetFullPath($Output)
    if ($Refit -and -not $Output.EndsWith('.refit.pdf',[StringComparison]::OrdinalIgnoreCase)) {
        throw 'Diagnostic refits require a separate .refit.pdf output'
    }
    $mode = if ($Refit) { 'refit' } else { 'normal' }
    $stage = Invoke-Guard $Python @('prepare','--mode',$mode,$Source,$Output)
    $stem = Split-Path $stage -Leaf
    $src = Join-Path $stage "$stem.pptx"
    $pdf = Join-Path $stage "$stem.pdf"
    $ppt = $null; $pres = $null; $mayBeOpen = $false
    try {
        $currentSession = (Get-Process -Id $PID).SessionId
        $existing = @(Get-Process -Name POWERPNT -ErrorAction SilentlyContinue | Where-Object SessionId -eq $currentSession)
        if ($existing.Count -gt 1) { throw 'Multiple PowerPoint instances; cannot establish single ownership' }
        if ($existing.Count -eq 1) { $ppt = [Runtime.InteropServices.Marshal]::GetActiveObject('PowerPoint.Application') }
        else { $ppt = New-Object -ComObject PowerPoint.Application }
        if ($ppt.Presentations.Count -ne 0) { throw 'PowerPoint already has an open presentation; stop' }
        $mayBeOpen = $true
        $pres = $ppt.Presentations.Open($src, -1, 0, 0)
        if ($pres.FullName -ne $src -or $ppt.Presentations.Count -ne 1) { throw 'Private presentation ownership could not be established' }
        $frames = @()
        $pptVersion = "$($ppt.Version) ($($ppt.Build))"
        foreach ($sl in $pres.Slides) {
            $frames += @(Get-Frames $sl.Shapes $sl.SlideIndex '')
            $sl.SlideShowTransition.Hidden = 0
        }
        if ($Refit) {
            foreach ($f in $frames) {
                if (-not $f.table -and $f.shape.TextFrame2.AutoSize -eq 2) {
                    $f.shape.TextFrame2.AutoSize = 0; $f.shape.TextFrame2.AutoSize = 2
                }
            }
        }
        $rows = @()
        foreach ($f in $frames) {
            $sh = $f.shape; $tf = $sh.TextFrame2; $tr = $tf.TextRange
            if ($tr.Length -eq 0) { continue }
            $paras = @()
            for ($k = 1; $k -le $tr.Paragraphs().Count; $k++) {
                $p = $tr.Paragraphs($k)
                $paras += [pscustomobject]@{ k = $k - 1; lines = $p.Lines().Count; size = $p.Font.Size; w = [math]::Round($p.BoundWidth, 1); text = $p.Text.Substring(0, [math]::Min(40, $p.Text.Length)) }
            }
            $rows += [pscustomobject]@{
                slide = $f.slide; id = $f.id; table = $f.table
                left = [math]::Round($sh.Left, 1); top = [math]::Round($sh.Top, 1); width = [math]::Round($sh.Width, 1); height = [math]::Round($sh.Height, 1)
                autosize = $tf.AutoSize; wrap = $tf.WordWrap
                mL = $tf.MarginLeft; mR = $tf.MarginRight; mT = $tf.MarginTop; mB = $tf.MarginBottom
                bl = [math]::Round($tr.BoundLeft, 1); bt = [math]::Round($tr.BoundTop, 1); bw = [math]::Round($tr.BoundWidth, 1); bh = [math]::Round($tr.BoundHeight, 1)
                lines = $tr.Lines().Count; paras = $paras
            }
        }
        $metrics = Join-Path $stage 'metrics.json'
        $json = ConvertTo-Json -InputObject ([pscustomobject]@{powerpoint=$pptVersion;frames=$rows}) -Depth 8
        [IO.File]::WriteAllText($metrics, $json, [Text.UTF8Encoding]::new($false))
        if ($ppt.Presentations.Count -ne 1 -or $pres.FullName -ne $src) { throw 'PowerPoint session changed' }
        if ($Refit) { $pres.SaveCopyAs((Join-Path $stage 'diagnostic.pptx')) }
        $pres.SaveAs($pdf,32)
        if (@($src,$pdf) -notcontains $pres.FullName) { throw 'Unknown document after PDF export' }
        $pres.Close(); $mayBeOpen = $false
        if ($ppt.Presentations.Count -ne 0) { throw 'Another presentation was opened' }
        Invoke-Guard $Python @('publish',$stage,'--office-version',$pptVersion) | Out-Null
        # Unique sidecars avoid overwriting another session's metrics or diagnostic deck.
        [IO.File]::Copy($metrics, "$Output.$stem.metrics.json", $false)
        if ($Refit) { [IO.File]::Copy((Join-Path $stage 'diagnostic.pptx'),"$Output.$stem.refit.pptx",$false) }
    } finally {
        if ($mayBeOpen -and $null -ne $pres -and @($src,$pdf) -contains $pres.FullName) {
            $pres.Close(); $mayBeOpen = $false
        }
        if ($null -ne $ppt) { [void][Runtime.InteropServices.Marshal]::ReleaseComObject($ppt) }
        if (-not $mayBeOpen) { Invoke-Guard $Python @('finish',$stage) | Out-Null }
        else { Write-Warning "Ownership uncertain; stage and lock retained: $stage" }
    }
}
Export-ModuleMember -Function Enter-RenderSession,Invoke-OwnedRender
