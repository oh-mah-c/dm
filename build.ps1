$CC = "gcc"
$CFLAGS = "-Iinclude", "-Iinclude/core", "-Isrc/core/dm_engine", "-Isrc/core/dm_engine/third_party/xla", "-Isrc/core/dm_engine/third_party/xla/third_party/tsl", "-DDM_NO_ICU", "-Wall", "-Wextra", "-O2"
$SRC_DIR = "src"
$OBJ_DIR = "obj"
$BIN_DIR = "bin"

# Create directories
New-Item -ItemType Directory -Force -Path $OBJ_DIR/core | Out-Null
New-Item -ItemType Directory -Force -Path $OBJ_DIR/algorithms | Out-Null
New-Item -ItemType Directory -Force -Path $BIN_DIR | Out-Null

$rspLines = Get-Content "build_dm.rsp" | Where-Object { $_ -match '\S' }
$objects = @()

foreach ($line in $rspLines) {
    $relPath = $line.Trim()
    $objPath = $relPath.Replace($SRC_DIR, $OBJ_DIR).Replace(".c", ".o")
    $parentDir = Split-Path $objPath -Parent
    if (-not (Test-Path $parentDir)) {
        New-Item -ItemType Directory -Force -Path $parentDir | Out-Null
    }
    
    $srcItem = Get-Item $relPath
    if (-not (Test-Path $objPath) -or ($srcItem.LastWriteTime -gt (Get-Item $objPath).LastWriteTime)) {
        Write-Host "Compiling $relPath..."
        & $CC @CFLAGS -c $relPath -o $objPath
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
    $objects += $objPath
}

Write-Host "Linking $BIN_DIR/dm.exe..."
& $CC $objects -o "$BIN_DIR/dm.exe" -lpsapi -lm
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Build successful!"
