# -----------------------------------------------------------------------------
# test/BH/CMakeLists.txt
# -----------------------------------------------------------------------------

file(RELATIVE_PATH _relPath "${PROJECT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "Adding index test found in ${_relPath}")

option(IPPL_ENABLE_OPENPMD "Enable openPMD output" OFF)

if(IPPL_ENABLE_OPENPMD)
    find_package(openPMD REQUIRED)
    set(BH_OPENPMD_LIBS openPMD::openPMD)
else()
    set(BH_OPENPMD_LIBS "")
endif()

if(IPPL_ENABLE_NBODY)
  if(CMAKE_CUDA_COMPILER OR CMAKE_HIP_COMPILER)
    set(_LANDAU_MGR LandauDampingBHManager.cu)
    set(_DIH_MGR    DisorderHeatingBHManager.cu)
    if(CMAKE_HIP_COMPILER AND NOT CMAKE_CUDA_COMPILER)
      set_source_files_properties(${_LANDAU_MGR} ${_DIH_MGR} PROPERTIES LANGUAGE HIP)
    endif()
  else()
    set(_LANDAU_MGR LandauDampingBHManager.cpp)
    set(_DIH_MGR    DisorderHeatingBHManager.cpp)
  endif()

  if(CMAKE_CUDA_COMPILER OR CMAKE_HIP_COMPILER)
    set(_PENNING_MGR PenningTrapBHManager.cu)
    if(CMAKE_HIP_COMPILER AND NOT CMAKE_CUDA_COMPILER)
      set_source_files_properties(${_PENNING_MGR} PROPERTIES LANGUAGE HIP)
    endif()
  else()
    set(_PENNING_MGR PenningTrapBHManager.cpp)
  endif()

  add_executable(LandauDampingBH
    LandauDampingBH.cpp
    ${_LANDAU_MGR})
  target_link_libraries(LandauDampingBH PRIVATE IPPL::ippl ippl::nbody ${BH_OPENPMD_LIBS})
  target_include_directories(LandauDampingBH PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

  add_executable(DisorderHeatingBH
    DisorderHeatingBH.cpp
    ${_DIH_MGR})
  target_link_libraries(DisorderHeatingBH PRIVATE IPPL::ippl ippl::nbody ${BH_OPENPMD_LIBS})
  target_include_directories(DisorderHeatingBH PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

  add_executable(PenningTrapBH
    PenningTrapBH.cpp
    ${_PENNING_MGR}
    ${_LANDAU_MGR})
  target_link_libraries(PenningTrapBH PRIVATE IPPL::ippl ippl::nbody ${BH_OPENPMD_LIBS})
  target_include_directories(PenningTrapBH PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

  # compile definitions — must come after add_executable
  if(IPPL_ENABLE_OPENPMD)
    target_compile_definitions(LandauDampingBH   PRIVATE IPPL_ENABLE_OPENPMD)
    target_compile_definitions(DisorderHeatingBH PRIVATE IPPL_ENABLE_OPENPMD)
    target_compile_definitions(PenningTrapBH     PRIVATE IPPL_ENABLE_OPENPMD)
  endif()

endif()