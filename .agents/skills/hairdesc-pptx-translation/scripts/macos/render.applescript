-- Native operations only. The Python launcher sends /usr/bin/open exactly once.
-- Never use active presentation, open/launch, quit, or save the private PPTX.
on locateDeck(sourcePath, pdfPath, sourceName, pdfName, allowPdf)
    set sourceHFS to (POSIX file sourcePath) as text
    set pdfHFS to (POSIX file pdfPath) as text
    log "EXPECTED name=" & sourceName & " posix=" & sourcePath & " hfs=" & sourceHFS
    set foundDeck to missing value
    set foundCount to 0
    set identityText to ""
    with timeout of 3 seconds
        if application "Microsoft PowerPoint" is not running then return {missing value, "ABSENT"}
        tell application "Microsoft PowerPoint"
            set deckCount to count of presentations
            log "PRESENTATION_COUNT=" & deckCount
            repeat with i from 1 to deckCount
                set d to presentation i
                set deckName to (get name of d) as text
                set deckPath to (get full name of d) as text
                set matchesSource to deckName is sourceName and (deckPath is sourcePath or deckPath is sourceHFS)
                log "SEEN name=" & deckName & " fullName=" & deckPath & " sourceMatch=" & matchesSource
                set matchesPdf to allowPdf and deckName is pdfName and (deckPath is pdfPath or deckPath is pdfHFS)
                if matchesSource or matchesPdf then
                    set foundDeck to d
                    set foundCount to foundCount + 1
                    set identityText to "FOUND" & tab & deckName & tab & deckPath
                end if
            end repeat
        end tell
    end timeout
    if foundCount > 1 then error "AMBIGUOUS_PRIVATE_PRESENTATION" number -2701
    if foundCount is 0 then return {missing value, "ABSENT"}
    return {foundDeck, identityText}
end locateDeck

on run argv
    if (count argv) is not 7 then error "Expected action source-path pdf-path source-name pdf-name allow-pdf font-probe"
    set actionName to item 1 of argv
    set sourcePath to item 2 of argv
    set pdfPath to item 3 of argv
    set sourceName to item 4 of argv
    set pdfName to item 5 of argv
    set allowPdf to item 6 of argv is "yes"
    set probeFonts to item 7 of argv is "yes"
    if actionName is "idle" then
        with timeout of 3 seconds
            if application "Microsoft PowerPoint" is not running then error "PowerPoint disappeared" number -2703
            tell application "Microsoft PowerPoint"
                if (count of presentations) is not 0 then error "PowerPoint has another presentation; stop" number -2703
            end tell
        end timeout
        return "IDLE"
    end if
    set found to my locateDeck(sourcePath, pdfPath, sourceName, pdfName, allowPdf)
    if actionName is "probe" then return item 2 of found
    set ownDoc to item 1 of found
    if ownDoc is missing value then error "PRIVATE_PRESENTATION_NOT_FOUND" number -2702
    if actionName is "export" then
        with timeout of 60 seconds
            tell application "Microsoft PowerPoint"
                save ownDoc in (POSIX file pdfPath) as save as PDF
            end tell
        end timeout
        return "EXPORTED" & tab & (item 2 of found)
    else if actionName is "close" then
        with timeout of 5 seconds
            tell application "Microsoft PowerPoint" to close ownDoc saving no
        end timeout
        return "CLOSED" & tab & (item 2 of found)
    end if
    error "Unknown action"
end run
