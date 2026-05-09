$CC = "gcc"
$CFLAGS = "-Iinclude", "-Iinclude/core", "-Wall", "-Wextra", "-O2"
$SRC_DIR = "src"
$OBJ_DIR = "obj"
$BIN_DIR = "bin"

# Create directories
New-Item -ItemType Directory -Force -Path $OBJ_DIR/core
New-Item -ItemType Directory -Force -Path $OBJ_DIR/algorithms
New-Item -ItemType Directory -Force -Path $BIN_DIR

# Get all source files
$sources = Get-ChildItem -Path $SRC_DIR -Filter *.c -Recurse
$objects = @()

foreach ($src in $sources) {
    $relPath = $src.FullName.Replace((Get-Location).Path + "\", "")
    $objPath = $relPath.Replace($SRC_DIR, $OBJ_DIR).Replace(".c", ".o")
    
    Write-Host "Compiling $relPath..."
    & $CC @CFLAGS -c $relPath -o $objPath
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    $objects += $objPath
}

Write-Host "Linking $BIN_DIR/dm.exe..."
& $CC $objects -o "$BIN_DIR/dm.exe" -lpsapi
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Build successful!"
