@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

echo Building BPE GPU...
cl.exe /DDM_GPU /nologo /W3 /O2 /D_CRT_SECURE_NO_WARNINGS ^
  /I include ^
  /I src\core\gpu\vulkan_hpp\Vulkan-Headers\include ^
  src\tools\dm_bpe.c ^
  src\tokenizer\bpe_subword.c ^
  src\core\gpu\dm_gpu.c ^
  src\core\gpu\gpu_bpe.c ^
  src\core\gpu\gpu_sinkhorn.c ^
  src\core\gpu\gpu_unigram_em.c ^
  /Fe:bin\dm_bpe_gpu.exe ^
  /link kernel32.lib

echo Building Unigram GPU...
cl.exe /DDM_GPU /nologo /W3 /O2 /D_CRT_SECURE_NO_WARNINGS ^
  /I include ^
  /I src\core\gpu\vulkan_hpp\Vulkan-Headers\include ^
  src\tools\dm_unigram.c ^
  src\tokenizer\unigram_subword.c ^
  src\core\gpu\dm_gpu.c ^
  src\core\gpu\gpu_bpe.c ^
  src\core\gpu\gpu_sinkhorn.c ^
  src\core\gpu\gpu_unigram_em.c ^
  /Fe:bin\dm_unigram_gpu.exe ^
  /link kernel32.lib

echo Building Volt GPU...
cl.exe /DDM_GPU /nologo /W3 /O2 /D_CRT_SECURE_NO_WARNINGS ^
  /I include ^
  /I src\core\gpu\vulkan_hpp\Vulkan-Headers\include ^
  src\tools\dm_volt.c ^
  src\tokenizer\volt.c ^
  src\core\gpu\dm_gpu.c ^
  src\core\gpu\gpu_bpe.c ^
  src\core\gpu\gpu_sinkhorn.c ^
  src\core\gpu\gpu_unigram_em.c ^
  /Fe:bin\dm_volt_gpu.exe ^
  /link kernel32.lib
