@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cd /d "e:\Folder_Code\dm"
echo Compiling dm_bpe with GPU support...
cl.exe /DDM_GPU /nologo /W3 /O2 /D_CRT_SECURE_NO_WARNINGS ^
  /I include ^
  /I src\core\gpu\Vulkan-Hpp\Vulkan-Headers\include ^
  src\tools\dm_bpe.c ^
  src\tokenizer\bpe_subword.c ^
  src\core\gpu\dm_gpu.c ^
  src\core\gpu\gpu_bpe.c ^
  src\core\gpu\gpu_sinkhorn.c ^
  src\core\gpu\gpu_unigram_em.c ^
  /Fe:bin\dm_bpe_gpu.exe ^
  /link kernel32.lib
if %ERRORLEVEL% neq 0 (
  echo BUILD FAILED with code %ERRORLEVEL%
  exit /b %ERRORLEVEL%
)
echo BUILD OK: bin\dm_bpe_gpu.exe
