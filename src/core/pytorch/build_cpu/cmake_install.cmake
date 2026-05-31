# Install script for directory: /home/autocookie/pomaieco/dm/src/core/pytorch

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/autocookie/pomaieco/dm/src/core/pytorch/dist")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/autocookie/pomaieco/dm/src/core/pytorch/build_cpu/third_party/protobuf/cmake/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/autocookie/pomaieco/dm/src/core/pytorch/build_cpu/confu-deps/pthreadpool/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/autocookie/pomaieco/dm/src/core/pytorch/build_cpu/confu-deps/cpuinfo/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/autocookie/pomaieco/dm/src/core/pytorch/build_cpu/confu-deps/pytorch_qnnpack/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/autocookie/pomaieco/dm/src/core/pytorch/build_cpu/confu-deps/NNPACK/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/autocookie/pomaieco/dm/src/core/pytorch/build_cpu/confu-deps/XNNPACK/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/autocookie/pomaieco/dm/src/core/pytorch/build_cpu/third_party/fbgemm/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/autocookie/pomaieco/dm/src/core/pytorch/build_cpu/third_party/ittapi/cmake_install.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/autocookie/pomaieco/dm/src/core/pytorch/dist/include")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/home/autocookie/pomaieco/dm/src/core/pytorch/dist" TYPE DIRECTORY MESSAGE_NEVER FILES "/home/autocookie/pomaieco/dm/src/core/pytorch/cmake/../third_party/pybind11/include" FILES_MATCHING REGEX "/[^/]*\\.h$")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/autocookie/pomaieco/dm/src/core/pytorch/build_cpu/third_party/ideep/mkl-dnn/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/autocookie/pomaieco/dm/src/core/pytorch/build_cpu/third_party/fmt/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/autocookie/pomaieco/dm/src/core/pytorch/build_cpu/torch/headeronly/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/autocookie/pomaieco/dm/src/core/pytorch/build_cpu/c10/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/autocookie/pomaieco/dm/src/core/pytorch/build_cpu/caffe2/cmake_install.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/cmake/Caffe2" TYPE FILE MESSAGE_NEVER FILES "/home/autocookie/pomaieco/dm/src/core/pytorch/build_cpu/Caffe2Config.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/cmake/Caffe2/public" TYPE FILE MESSAGE_NEVER FILES
    "/home/autocookie/pomaieco/dm/src/core/pytorch/cmake/public/cuda.cmake"
    "/home/autocookie/pomaieco/dm/src/core/pytorch/cmake/public/xpu.cmake"
    "/home/autocookie/pomaieco/dm/src/core/pytorch/cmake/public/glog.cmake"
    "/home/autocookie/pomaieco/dm/src/core/pytorch/cmake/public/gflags.cmake"
    "/home/autocookie/pomaieco/dm/src/core/pytorch/cmake/public/mkl.cmake"
    "/home/autocookie/pomaieco/dm/src/core/pytorch/cmake/public/mkldnn.cmake"
    "/home/autocookie/pomaieco/dm/src/core/pytorch/cmake/public/protobuf.cmake"
    "/home/autocookie/pomaieco/dm/src/core/pytorch/cmake/public/utils.cmake"
    "/home/autocookie/pomaieco/dm/src/core/pytorch/cmake/public/LoadHIP.cmake"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/cmake/Caffe2/" TYPE DIRECTORY MESSAGE_NEVER FILES "/home/autocookie/pomaieco/dm/src/core/pytorch/cmake/Modules_CUDA_fix")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/cmake/Caffe2" TYPE FILE MESSAGE_NEVER FILES "/home/autocookie/pomaieco/dm/src/core/pytorch/cmake/Modules/FindCUDAToolkit.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/cmake/Caffe2" TYPE FILE MESSAGE_NEVER FILES "/home/autocookie/pomaieco/dm/src/core/pytorch/cmake/Modules/FindCUSPARSELT.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/cmake/Caffe2" TYPE FILE MESSAGE_NEVER FILES "/home/autocookie/pomaieco/dm/src/core/pytorch/cmake/Modules/FindCUDSS.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/cmake/Caffe2" TYPE FILE MESSAGE_NEVER FILES "/home/autocookie/pomaieco/dm/src/core/pytorch/cmake/Modules/FindSYCLToolkit.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/cmake/Caffe2/Caffe2Targets.cmake")
    file(DIFFERENT _cmake_export_file_changed FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/cmake/Caffe2/Caffe2Targets.cmake"
         "/home/autocookie/pomaieco/dm/src/core/pytorch/build_cpu/CMakeFiles/Export/660a2c44cf5c98167ccb9da2e0f32625/Caffe2Targets.cmake")
    if(_cmake_export_file_changed)
      file(GLOB _cmake_old_config_files "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/cmake/Caffe2/Caffe2Targets-*.cmake")
      if(_cmake_old_config_files)
        string(REPLACE ";" ", " _cmake_old_config_files_text "${_cmake_old_config_files}")
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/cmake/Caffe2/Caffe2Targets.cmake\" will be replaced.  Removing files [${_cmake_old_config_files_text}].")
        unset(_cmake_old_config_files_text)
        file(REMOVE ${_cmake_old_config_files})
      endif()
      unset(_cmake_old_config_files)
    endif()
    unset(_cmake_export_file_changed)
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/cmake/Caffe2" TYPE FILE MESSAGE_NEVER FILES "/home/autocookie/pomaieco/dm/src/core/pytorch/build_cpu/CMakeFiles/Export/660a2c44cf5c98167ccb9da2e0f32625/Caffe2Targets.cmake")
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/cmake/Caffe2" TYPE FILE MESSAGE_NEVER FILES "/home/autocookie/pomaieco/dm/src/core/pytorch/build_cpu/CMakeFiles/Export/660a2c44cf5c98167ccb9da2e0f32625/Caffe2Targets-release.cmake")
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/autocookie/pomaieco/dm/src/core/pytorch/torchgen/packaged/ATen/native/native_functions.yaml;/home/autocookie/pomaieco/dm/src/core/pytorch/torchgen/packaged/ATen/native/tags.yaml")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/home/autocookie/pomaieco/dm/src/core/pytorch/torchgen/packaged/ATen/native" TYPE FILE MESSAGE_NEVER FILES
    "/home/autocookie/pomaieco/dm/src/core/pytorch/aten/src/ATen/native/native_functions.yaml"
    "/home/autocookie/pomaieco/dm/src/core/pytorch/aten/src/ATen/native/tags.yaml"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/autocookie/pomaieco/dm/src/core/pytorch/torchgen/packaged/ATen/templates/")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/home/autocookie/pomaieco/dm/src/core/pytorch/torchgen/packaged/ATen/templates" TYPE DIRECTORY MESSAGE_NEVER FILES "/home/autocookie/pomaieco/dm/src/core/pytorch/aten/src/ATen/templates/")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/autocookie/pomaieco/dm/src/core/pytorch/torchgen/packaged/autograd/")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/home/autocookie/pomaieco/dm/src/core/pytorch/torchgen/packaged/autograd" TYPE DIRECTORY MESSAGE_NEVER FILES "/home/autocookie/pomaieco/dm/src/core/pytorch/tools/autograd/" REGEX "/BUILD\\.bazel$" EXCLUDE REGEX "/[^/]*\\.bzl$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/autocookie/pomaieco/dm/src/core/pytorch/torch/_inductor/kernel/vendored_templates/cutedsl/kernels/cutedsl_grouped_gemm.py")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/home/autocookie/pomaieco/dm/src/core/pytorch/torch/_inductor/kernel/vendored_templates/cutedsl/kernels" TYPE FILE MESSAGE_NEVER RENAME "cutedsl_grouped_gemm.py" FILES "/home/autocookie/pomaieco/dm/src/core/pytorch/third_party/cutlass/examples/python/CuTeDSL/blackwell/grouped_gemm.py")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/autocookie/pomaieco/dm/src/core/pytorch/torch/_inductor/kernel/vendored_templates/cutedsl/kernels/__init__.py")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/home/autocookie/pomaieco/dm/src/core/pytorch/torch/_inductor/kernel/vendored_templates/cutedsl/kernels" TYPE FILE MESSAGE_NEVER RENAME "__init__.py" FILES "/home/autocookie/pomaieco/dm/src/core/pytorch/build_cpu/_empty_init.py")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/autocookie/pomaieco/dm/src/core/pytorch/tools/shared/_utils_internal.py")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/home/autocookie/pomaieco/dm/src/core/pytorch/tools/shared" TYPE FILE MESSAGE_NEVER FILES "/home/autocookie/pomaieco/dm/src/core/pytorch/torch/_utils_internal.py")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/autocookie/pomaieco/dm/src/core/pytorch/torch/utils/benchmark/utils/valgrind_wrapper/callgrind.h")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/home/autocookie/pomaieco/dm/src/core/pytorch/torch/utils/benchmark/utils/valgrind_wrapper" TYPE FILE MESSAGE_NEVER FILES "/home/autocookie/pomaieco/dm/src/core/pytorch/third_party/valgrind-headers/callgrind.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/autocookie/pomaieco/dm/src/core/pytorch/torch/utils/benchmark/utils/valgrind_wrapper/valgrind.h")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/home/autocookie/pomaieco/dm/src/core/pytorch/torch/utils/benchmark/utils/valgrind_wrapper" TYPE FILE MESSAGE_NEVER FILES "/home/autocookie/pomaieco/dm/src/core/pytorch/third_party/valgrind-headers/valgrind.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/." TYPE DIRECTORY MESSAGE_NEVER FILES "/home/autocookie/pomaieco/dm/src/core/pytorch/torch/" FILES_MATCHING REGEX "/[^/]*\\.pyi$" REGEX "/py\\.typed$")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/utils/benchmark/utils" TYPE DIRECTORY MESSAGE_NEVER FILES "/home/autocookie/pomaieco/dm/src/core/pytorch/torch/utils/benchmark/utils/" FILES_MATCHING REGEX "/[^/]*\\.cpp$" REGEX "/[^/]*\\.h$")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/utils/model_dump" TYPE FILE OPTIONAL MESSAGE_NEVER FILES
    "/home/autocookie/pomaieco/dm/src/core/pytorch/torch/utils/model_dump/skeleton.html"
    "/home/autocookie/pomaieco/dm/src/core/pytorch/torch/utils/model_dump/code.js"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/utils/model_dump" TYPE DIRECTORY MESSAGE_NEVER FILES "/home/autocookie/pomaieco/dm/src/core/pytorch/torch/utils/model_dump/" FILES_MATCHING REGEX "/[^/]*\\.mjs$")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/_inductor" TYPE FILE OPTIONAL MESSAGE_NEVER FILES "/home/autocookie/pomaieco/dm/src/core/pytorch/torch/_inductor/script.ld")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/_inductor/codegen" TYPE DIRECTORY MESSAGE_NEVER FILES "/home/autocookie/pomaieco/dm/src/core/pytorch/torch/_inductor/codegen/" FILES_MATCHING REGEX "/[^/]*\\.h$" REGEX "/[^/]*\\.cpp$")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/_inductor/kernel/flex/templates" TYPE DIRECTORY MESSAGE_NEVER FILES "/home/autocookie/pomaieco/dm/src/core/pytorch/torch/_inductor/kernel/flex/templates/" FILES_MATCHING REGEX "/[^/]*\\.jinja$")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/_inductor/kernel/templates" TYPE DIRECTORY MESSAGE_NEVER FILES "/home/autocookie/pomaieco/dm/src/core/pytorch/torch/_inductor/kernel/templates/" FILES_MATCHING REGEX "/[^/]*\\.jinja$")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/_export/serde" TYPE DIRECTORY MESSAGE_NEVER FILES "/home/autocookie/pomaieco/dm/src/core/pytorch/torch/_export/serde/" FILES_MATCHING REGEX "/[^/]*\\.yaml$" REGEX "/[^/]*\\.thrift$")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/csrc/inductor/aoti_runtime" TYPE FILE OPTIONAL MESSAGE_NEVER FILES "/home/autocookie/pomaieco/dm/src/core/pytorch/torch/csrc/inductor/aoti_runtime/model.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/testing/_internal/generated" TYPE FILE MESSAGE_NEVER FILES "/home/autocookie/pomaieco/dm/src/core/pytorch/torch/testing/_internal/generated/annotated_fn_args.py")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/_dynamo" TYPE FILE OPTIONAL MESSAGE_NEVER FILES "/home/autocookie/pomaieco/dm/src/core/pytorch/torch/_dynamo/graph_break_registry.json")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  
  execute_process(
    COMMAND "/usr/bin/python3"
      "/home/autocookie/pomaieco/dm/src/core/pytorch/tools/wrap_headers.py"
      "${CMAKE_INSTALL_PREFIX}/include"
    COMMAND_ERROR_IS_FATAL ANY
  )

endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/home/autocookie/pomaieco/dm/src/core/pytorch/build_cpu/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
