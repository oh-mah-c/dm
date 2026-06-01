# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/autocookie/pomaieco/dm/src/core/pytorch/build_cpu/confu-srcs/six"
  "/home/autocookie/pomaieco/dm/src/core/pytorch/build_cpu/confu-deps/six"
  "/home/autocookie/pomaieco/dm/src/core/pytorch/build_cpu/confu-deps/six-download/six-prefix"
  "/home/autocookie/pomaieco/dm/src/core/pytorch/build_cpu/confu-deps/six-download/six-prefix/tmp"
  "/home/autocookie/pomaieco/dm/src/core/pytorch/build_cpu/confu-deps/six-download/six-prefix/src/six-stamp"
  "/home/autocookie/pomaieco/dm/src/core/pytorch/build_cpu/confu-deps/six-download/six-prefix/src"
  "/home/autocookie/pomaieco/dm/src/core/pytorch/build_cpu/confu-deps/six-download/six-prefix/src/six-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/autocookie/pomaieco/dm/src/core/pytorch/build_cpu/confu-deps/six-download/six-prefix/src/six-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/autocookie/pomaieco/dm/src/core/pytorch/build_cpu/confu-deps/six-download/six-prefix/src/six-stamp${cfgdir}") # cfgdir has leading slash
endif()
