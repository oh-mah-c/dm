# Install script for directory: /home/autocookie/pomaieco/dm/src/core/pytorch/third_party/fbgemm

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

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY MESSAGE_NEVER FILES "/home/autocookie/pomaieco/dm/src/core/pytorch/build_cpu/lib/libfbgemm.a")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/fbgemm" TYPE FILE MESSAGE_NEVER FILES
    "/home/autocookie/pomaieco/dm/src/core/pytorch/third_party/fbgemm/include/fbgemm/Assert.h"
    "/home/autocookie/pomaieco/dm/src/core/pytorch/third_party/fbgemm/include/fbgemm/ConvUtils.h"
    "/home/autocookie/pomaieco/dm/src/core/pytorch/third_party/fbgemm/include/fbgemm/Fbgemm.h"
    "/home/autocookie/pomaieco/dm/src/core/pytorch/third_party/fbgemm/include/fbgemm/FbgemmBuild.h"
    "/home/autocookie/pomaieco/dm/src/core/pytorch/third_party/fbgemm/include/fbgemm/FbgemmConvert.h"
    "/home/autocookie/pomaieco/dm/src/core/pytorch/third_party/fbgemm/include/fbgemm/FbgemmEmbedding.h"
    "/home/autocookie/pomaieco/dm/src/core/pytorch/third_party/fbgemm/include/fbgemm/FbgemmFP16.h"
    "/home/autocookie/pomaieco/dm/src/core/pytorch/third_party/fbgemm/include/fbgemm/FbgemmFP32.h"
    "/home/autocookie/pomaieco/dm/src/core/pytorch/third_party/fbgemm/include/fbgemm/FbgemmFPCommon.h"
    "/home/autocookie/pomaieco/dm/src/core/pytorch/third_party/fbgemm/include/fbgemm/FbgemmI64.h"
    "/home/autocookie/pomaieco/dm/src/core/pytorch/third_party/fbgemm/include/fbgemm/FbgemmI8DepthwiseAvx2.h"
    "/home/autocookie/pomaieco/dm/src/core/pytorch/third_party/fbgemm/include/fbgemm/FbgemmI8DirectconvAvx2.h"
    "/home/autocookie/pomaieco/dm/src/core/pytorch/third_party/fbgemm/include/fbgemm/FbgemmI8Spmdm.h"
    "/home/autocookie/pomaieco/dm/src/core/pytorch/third_party/fbgemm/include/fbgemm/FbgemmPackMatrixB.h"
    "/home/autocookie/pomaieco/dm/src/core/pytorch/third_party/fbgemm/include/fbgemm/FbgemmSparse.h"
    "/home/autocookie/pomaieco/dm/src/core/pytorch/third_party/fbgemm/include/fbgemm/FloatConversion.h"
    "/home/autocookie/pomaieco/dm/src/core/pytorch/third_party/fbgemm/include/fbgemm/OutputProcessing-inl.h"
    "/home/autocookie/pomaieco/dm/src/core/pytorch/third_party/fbgemm/include/fbgemm/PackingTraits-inl.h"
    "/home/autocookie/pomaieco/dm/src/core/pytorch/third_party/fbgemm/include/fbgemm/QuantUtils.h"
    "/home/autocookie/pomaieco/dm/src/core/pytorch/third_party/fbgemm/include/fbgemm/QuantUtilsAvx2.h"
    "/home/autocookie/pomaieco/dm/src/core/pytorch/third_party/fbgemm/include/fbgemm/QuantUtilsAvx512.h"
    "/home/autocookie/pomaieco/dm/src/core/pytorch/third_party/fbgemm/include/fbgemm/QuantUtilsNeon.h"
    "/home/autocookie/pomaieco/dm/src/core/pytorch/third_party/fbgemm/include/fbgemm/spmmUtils.h"
    "/home/autocookie/pomaieco/dm/src/core/pytorch/third_party/fbgemm/include/fbgemm/spmmUtilsAvx2.h"
    "/home/autocookie/pomaieco/dm/src/core/pytorch/third_party/fbgemm/include/fbgemm/SimdUtils.h"
    "/home/autocookie/pomaieco/dm/src/core/pytorch/third_party/fbgemm/include/fbgemm/Utils.h"
    "/home/autocookie/pomaieco/dm/src/core/pytorch/third_party/fbgemm/include/fbgemm/UtilsAvx2.h"
    "/home/autocookie/pomaieco/dm/src/core/pytorch/third_party/fbgemm/include/fbgemm/Types.h"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/cmake/fbgemm/fbgemmLibraryConfig.cmake")
    file(DIFFERENT _cmake_export_file_changed FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/cmake/fbgemm/fbgemmLibraryConfig.cmake"
         "/home/autocookie/pomaieco/dm/src/core/pytorch/build_cpu/third_party/fbgemm/CMakeFiles/Export/216576ddd81235114a741aee81201340/fbgemmLibraryConfig.cmake")
    if(_cmake_export_file_changed)
      file(GLOB _cmake_old_config_files "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/cmake/fbgemm/fbgemmLibraryConfig-*.cmake")
      if(_cmake_old_config_files)
        string(REPLACE ";" ", " _cmake_old_config_files_text "${_cmake_old_config_files}")
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/cmake/fbgemm/fbgemmLibraryConfig.cmake\" will be replaced.  Removing files [${_cmake_old_config_files_text}].")
        unset(_cmake_old_config_files_text)
        file(REMOVE ${_cmake_old_config_files})
      endif()
      unset(_cmake_old_config_files)
    endif()
    unset(_cmake_export_file_changed)
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/cmake/fbgemm" TYPE FILE MESSAGE_NEVER FILES "/home/autocookie/pomaieco/dm/src/core/pytorch/build_cpu/third_party/fbgemm/CMakeFiles/Export/216576ddd81235114a741aee81201340/fbgemmLibraryConfig.cmake")
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/cmake/fbgemm" TYPE FILE MESSAGE_NEVER FILES "/home/autocookie/pomaieco/dm/src/core/pytorch/build_cpu/third_party/fbgemm/CMakeFiles/Export/216576ddd81235114a741aee81201340/fbgemmLibraryConfig-release.cmake")
  endif()
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/autocookie/pomaieco/dm/src/core/pytorch/build_cpu/third_party/fbgemm/asmjit/cmake_install.cmake")
endif()

