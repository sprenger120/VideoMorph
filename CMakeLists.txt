cmake_minimum_required(VERSION 3.30)
project(emojify)

set(CMAKE_CXX_STANDARD 23)
find_package(OpenCV REQUIRED)


add_executable(emojify src/main.cpp)
target_include_directories(${PROJECT_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})


