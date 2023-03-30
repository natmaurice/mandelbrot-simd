cmake_minimum_required(VERSION 3.10)

set(ProjectName mandelbrot)
project(mandelbrot VERSION 0.1
  LANGUAGES C)

set(src_dir ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(cuda_dir ${CMAKE_CURRENT_SOURCE_DIR}/cuda)
set(inc_dir ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(lib_dir ${CMAKE_CURRENT_SOURCE_DIR}/lib)

set(exec mandelbrot)

LIST(APPEND mandelbrot_src_files
  ${src_dir}/main.c
  ${src_dir}/mandelbrot.c
  ${src_dir}/mandelbrot-avx512.c
)

if(MANDELBROT_USE_AVX512)
  LIST(APPEND mandelbrot_src_files
    ${src_dir}/mandelbrot-avx512.c
  )
endif()

if(MANDELBROT_USE_CUDA)
  LIST(APPEND mandelbrot_src_files
    ${cuda_dir}/mandelbrot.cu
  )

endif()

add_subdirectory(${lib_dir}/nrc2)

find_package(OpenMP REQUIRED)
add_compile_options("${OpenMP_C_FLAGS}")


add_executable(${exec} ${mandelbrot_src_files})


target_include_directories(${exec} PRIVATE ${inc_dir})
if(MANDELBROT_USE_CUDA)
  target_include_directories(${exec} PUBLIC ${cuda_dir})
  target_compile_definitions(${exec} PUBLIC -DMANDELBROT_USE_CUDA)
endif()


target_link_libraries(${exec} PUBLIC nrc-slib ${OpenMP_C_LIBRARIES})
