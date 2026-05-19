param(
    [string]$OutDir = "results/hiep_q1"
)

$ErrorActionPreference = "Stop"
$RootDir = Resolve-Path (Join-Path $PSScriptRoot "..")
$BinDir = Join-Path $RootDir "bin"
$Bin = Join-Path $BinDir "dm.exe"

New-Item -ItemType Directory -Force -Path $BinDir | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $RootDir $OutDir) | Out-Null

Push-Location $RootDir
try {
    $sources = @("src\main.c")
    $sources += (Get-ChildItem "src\core" -Filter "*.c" | ForEach-Object { $_.FullName })
    $sources += "src\tokenizer\faro_tokenizer.c", "src\tokenizer\tokenizer_variants.c"
    $sources += (Get-ChildItem "src\algorithms" -Filter "*.c" | ForEach-Object { $_.FullName })

    gcc -Iinclude -Iinclude/core -Wall -Wextra -O2 -pthread $sources -o $Bin -lm -lpsapi

    python scripts\run_hiep_experiments.py $OutDir $Bin $RootDir
    python scripts\plot_hiep_results.py $OutDir
    python scripts\run_hiep_baseline_experiments.py $OutDir $Bin $RootDir 35
} finally {
    Pop-Location
}
