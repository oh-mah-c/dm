# compile_shaders.ps1
# Compile all DM GPU compute shaders from GLSL to SPIR-V.
#
# Uses glslc from the Vulkan SDK (if installed) or from the glslang submodule.
# Output goes to bin/shaders/ so executables can find them at runtime.
#
# Usage:
#   .\scripts\compile_shaders.ps1
#   .\scripts\compile_shaders.ps1 -OutputDir path\to\shaders
#   .\scripts\compile_shaders.ps1 -Verbose

param(
    [string]$OutputDir = "bin\shaders"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ── Locate Compiler ────────────────────────────────────────────────────────────

function Find-ShaderCompiler {
    # 1. Vulkan SDK (standard install)
    $sdk = $env:VULKAN_SDK
    if ($sdk) {
        $candidate = Join-Path $sdk "Bin\glslc.exe"
        if (Test-Path $candidate) { return @{ Path = $candidate; Type = "glslc" } }
        $candidate2 = Join-Path $sdk "Bin\glslangValidator.exe"
        if (Test-Path $candidate2) { return @{ Path = $candidate2; Type = "glslang" } }
    }
    # 2. glslang submodule (needs to be compiled first)
    $sub = "third_party\Vulkan-Hpp\glslang\build\StandAlone\Release\glslc.exe"
    if (Test-Path $sub) { return @{ Path = $sub; Type = "glslc" } }
    $sub2 = "third_party\Vulkan-Hpp\glslang\build\StandAlone\Release\glslangValidator.exe"
    if (Test-Path $sub2) { return @{ Path = $sub2; Type = "glslang" } }
    # 3. PATH
    $inPath = Get-Command glslc -ErrorAction SilentlyContinue
    if ($inPath) { return @{ Path = $inPath.Source; Type = "glslc" } }
    $inPath2 = Get-Command glslangValidator -ErrorAction SilentlyContinue
    if ($inPath2) { return @{ Path = $inPath2.Source; Type = "glslang" } }
    return $null
}

$compiler = Find-ShaderCompiler
if (-not $compiler) {
    Write-Error @"
glslc or glslangValidator not found.  Install one of:
  A) Vulkan SDK: https://vulkan.lunarg.com/sdk/home  (sets VULKAN_SDK env var)
  B) Build glslang submodule:
       cmake -S third_party/Vulkan-Hpp/glslang -B third_party/Vulkan-Hpp/glslang/build
       cmake --build third_party/Vulkan-Hpp/glslang/build --config Release
"@
    exit 1
}
Write-Host "Compiler: $($compiler.Path) ($($compiler.Type))"

# ── Shader list ───────────────────────────────────────────────────────────────

$ShaderDir = "src\gpu\shaders"
$shaders = @(
    "bpe_pair_count.glsl",
    "sinkhorn_spmv.glsl",
    "sinkhorn_rowsum.glsl",
    "unigram_word.glsl"
)

# ── Create output directory ───────────────────────────────────────────────────

if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir | Out-Null
}

# ── Compile ───────────────────────────────────────────────────────────────────

$errors = 0
foreach ($shader in $shaders) {
    $src  = Join-Path $ShaderDir $shader
    $name = [System.IO.Path]::GetFileNameWithoutExtension($shader)
    $dst  = Join-Path $OutputDir "$name.spv"

    if (-not (Test-Path $src)) {
        Write-Warning "Source not found: $src"
        $errors++
        continue
    }

    Write-Host "Compiling $shader -> $dst"
    if ($compiler.Type -eq "glslc") {
        $proc = Start-Process -FilePath $compiler.Path `
            -ArgumentList "--target-env=vulkan1.1", "-O", $src, "-o", $dst `
            -NoNewWindow -Wait -PassThru
    } else {
        $proc = Start-Process -FilePath $compiler.Path `
            -ArgumentList "-V", "-S", "comp", "--target-env", "vulkan1.1", $src, "-o", $dst `
            -NoNewWindow -Wait -PassThru
    }
    if ($proc.ExitCode -ne 0) {
        Write-Error "Failed: $shader (exit $($proc.ExitCode))"
        $errors++
    }
}

if ($errors -gt 0) {
    Write-Error "$errors shader(s) failed to compile."
    exit 1
}

Write-Host ""
Write-Host "All shaders compiled to $OutputDir"
Write-Host "Run tokenizers with --gpu to use GPU acceleration."
