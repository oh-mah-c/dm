@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
"C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" -S "e:\Folder_Code\dm\third_party\Vulkan-Hpp\glslang" -B "e:\Folder_Code\dm\third_party\Vulkan-Hpp\glslang\build" -DCMAKE_BUILD_TYPE=Release -DENABLE_GLSLANG_BINARIES=ON -DENABLE_SPVREMAPPER=OFF -DENABLE_HLSL=OFF -DBUILD_TESTING=OFF -DENABLE_OPT=OFF 2>&1
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%
"C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" --build "e:\Folder_Code\dm\third_party\Vulkan-Hpp\glslang\build" --config Release --target glslang-standalone -- /m 2>&1
