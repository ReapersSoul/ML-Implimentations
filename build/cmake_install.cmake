# Install script for directory: /media/trevor/49AD9A774ADB16B6/Echotones/Software/ML Implimentations

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Debug")
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
  # Include the install script for each subdirectory.
  include("/media/trevor/49AD9A774ADB16B6/Echotones/Software/ML Implimentations/build/MLLib/cmake_install.cmake")
  include("/media/trevor/49AD9A774ADB16B6/Echotones/Software/ML Implimentations/build/1-Perceptron/cmake_install.cmake")
  include("/media/trevor/49AD9A774ADB16B6/Echotones/Software/ML Implimentations/build/2-PerceptronLayer/cmake_install.cmake")
  include("/media/trevor/49AD9A774ADB16B6/Echotones/Software/ML Implimentations/build/3-DeepNeuralNetwork/cmake_install.cmake")
  include("/media/trevor/49AD9A774ADB16B6/Echotones/Software/ML Implimentations/build/4-RecurrentPerceptron/cmake_install.cmake")
  include("/media/trevor/49AD9A774ADB16B6/Echotones/Software/ML Implimentations/build/5-RecurrentLayer/cmake_install.cmake")
  include("/media/trevor/49AD9A774ADB16B6/Echotones/Software/ML Implimentations/build/6-RecurrentNeuralNetwork/cmake_install.cmake")
  include("/media/trevor/49AD9A774ADB16B6/Echotones/Software/ML Implimentations/build/7-2DValidConvolutionLayer/cmake_install.cmake")
  include("/media/trevor/49AD9A774ADB16B6/Echotones/Software/ML Implimentations/build/8-2DFullConvolutionLayer/cmake_install.cmake")

endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/media/trevor/49AD9A774ADB16B6/Echotones/Software/ML Implimentations/build/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
