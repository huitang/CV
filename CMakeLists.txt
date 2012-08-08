project(CV)

cmake_minimum_required(VERSION 2.6)

find_package(ITK)
if(ITK_FOUND)
  include(${ITK_USE_FILE})
else(ITK_FOUND)
  message(FATAL_ERROR
          "Cannot build RSFTest without ITK.  Please set ITK_DIR.")
endif(ITK_FOUND)

if( WIN32 )
  SET( CMAKE_CXX_FLAGS -DDLL_EXPORTS)
endif()

set(LevelSetsSourceList
  CVTermTest
)

foreach( var ${LevelSetsSourceList} )
  add_executable(${var} ${var}/${var}.cxx)
  target_link_libraries(${var} ${ITK_LIBRARIES}  )
endforeach()

include( CTest )

