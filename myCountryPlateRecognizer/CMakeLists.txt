cmake_minimum_required(VERSION 2.8.12)
# Every project needs a name.  We call this the "examples" project.
project(myPlatesCountryRecognitions)

set (CMAKE_CXX_STANDARD 11)

# need for static Tesseract
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# need for static Tesseract
find_package(OpenMP)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

#SET(CMAKE_PREFIX_PATH "/opencv341-CUDA")
#find_package(OpenCV PATHS "/opencv341-CUDA")

# set(Boost_USE_STATIC_LIBS ON) /* turned off for -lboost_locale */
#set(Boost_USE_MULTITHREADED ON)
#set(Boost_USE_STATIC_RUNTIME ON)
#find_package(Boost COMPONENTS filesystem system locale REQUIRED)
#find_package(OpenCV REQUIRED )

# Tell cmake we will need dlib.  This command will pull in dlib and compile it
# into your project.  Note that you don't need to compile or install dlib.  All
# cmake needs is the dlib source code folder and it will take care of everything.
add_subdirectory(../dlib dlib_build)


# The next thing we need to do is tell CMake about the code you want to
# compile.  We do this with the add_executable() statement which takes the name
# of the output executable and then a list of .cpp files to compile.  Here we
# are going to compile one of the dlib example programs which has only one .cpp
# file, assignment_learning_ex.cpp.  If your program consisted of multiple .cpp
# files you would simply list them here in the add_executable() statement.
## add_executable(assignment_learning_ex assignment_learning_ex.cpp)
# Finally, you need to tell CMake that this program, assignment_learning_ex,
# depends on dlib.  You do that with this statement:
##target_link_libraries(assignment_learning_ex dlib::dlib)



# To compile this program all you need to do is ask cmake.  You would type
# these commands from within the directory containing this CMakeLists.txt
# file:
#   mkdir build
#   cd build
#   cmake ..
#   cmake --build . --config Release
#
# The cmake .. command looks in the parent folder for a file named
# CMakeLists.txt, reads it, and sets up everything needed to build program.
# Also, note that CMake can generate Visual Studio or XCode project files.  So
# if instead you had written:
#   cd build
#   cmake .. -G Xcode
#
# You would be able to open the resulting Xcode project and compile and edit
# the example programs within the Xcode IDE.  CMake can generate a lot of
# different types of IDE projects.  Run the cmake -h command to see a list of
# arguments to -G to see what kinds of projects cmake can generate for you.  It
# probably includes your favorite IDE in the list.




#################################################################################
#################################################################################
#  A CMakeLists.txt file can compile more than just one program.  So below we
#  tell it to compile the other dlib example programs using pretty much the
#  same CMake commands we used above.
#################################################################################
#################################################################################


# Since there are a lot of examples I'm going to use a macro to simplify this
# CMakeLists.txt file.  However, usually you will create only one executable in
# your cmake projects and use the syntax shown above.

set(HEADER_FILES ../myPlatesDetector/myUtils.h)

macro(add_example name)
   add_executable(${name} ${name}.cpp ${HEADER_FILES})
   target_link_libraries(${name} dlib::dlib )
   target_link_libraries(${name} ${OpenCV_LIBS} )
   target_link_libraries(${name} Threads::Threads)
endmacro()

# if an example requires GUI, call this macro to check DLIB_NO_GUI_SUPPORT to include or exclude
macro(add_gui_example name)
   if (DLIB_NO_GUI_SUPPORT)
      message("No GUI support, so we won't build the ${name} example.")
   else()
      add_example(${name})
   endif()
endmacro()

# The deep learning toolkit requires a compiler with essentially complete C++11
# support.  However, versions of Visual Studio prior to October 2016 didn't
# provide enough C++11 support to compile the DNN tooling, but were good enough
# to compile the rest of dlib.  So new versions of Visual Studio 2015 will
# work.  However, Visual Studio 2017 had some C++11 support regressions, so it
# wasn't until December 2017 that Visual Studio 2017 had good enough C++11
# support to compile the DNN examples.  So if you are using Visual Studio, make
# sure you have an updated version if you want to compile the DNN code.
#
# Also note that Visual Studio users should give the -T host=x64 option so that
# CMake will instruct Visual Studio to use its 64bit toolchain.  If you don't
# do this then by default Visual Studio uses a 32bit toolchain, WHICH RESTRICTS
# THE COMPILER TO ONLY 2GB OF RAM, causing it to run out of RAM and crash when
# compiling some of the DNN examples.  So generate your project with a statement
# like this:
#   cmake .. -G "Visual Studio 14 2015 Win64" -T host=x64
if (NOT USING_OLD_VISUAL_STUDIO_COMPILER)
   add_gui_example(dnn_countryRecognition)
   add_gui_example(dnn_countryRecognition-v2)
   add_gui_example(dnn_countryRecognition-v3)
   add_gui_example(dnn_countryRecognition-v4)
   add_gui_example(dnn_countryRec_test)
   add_gui_example(dnn_co_multi_rec_test)
   #add_gui_example(dnn_mmod_train_find_plates_model5)
   #add_gui_example(dnn_mmod_train_find_plates_model5-skip)
   # add_example(dnn_metric_learning_ex)
   # add_gui_example(dnn_face_recognition_ex)
   # add_example(dnn_introduction_ex)
   # add_example(dnn_introduction2_ex)
   # add_example(dnn_inception_ex)
   # add_gui_example(dnn_mmod_ex)
   # add_gui_example(dnn_mmod_face_detection_ex)
   # add_gui_example(random_cropper_ex)
   # add_gui_example(dnn_mmod_dog_hipsterizer)
   # add_gui_example(dnn_imagenet_ex)
   #add_gui_example(dnn_mmod_find_cars_ex)
   #add_gui_example(dnn_mmod_find_cars2_ex)
   # add_example(dnn_mmod_train_find_cars_ex)
   #add_gui_example(dnn_semantic_segmentation_ex)
   # add_example(dnn_imagenet_train_ex)
   #add_example(dnn_semantic_segmentation_train_ex)
   # add_example(dnn_metric_learning_on_images_ex)
   #add_gui_example(train_shape_predictor_ex)
endif()


if (DLIB_NO_GUI_SUPPORT)
   message("No GUI support, so we won't build the webcam_face_pose_ex example.")
else()
   find_package(OpenCV QUIET)
   if (OpenCV_FOUND)
      include_directories(${OpenCV_INCLUDE_DIRS})

      # add_executable(webcam_face_pose_ex webcam_face_pose_ex.cpp)
      # target_link_libraries(webcam_face_pose_ex dlib::dlib ${OpenCV_LIBS} )
   else()
      message("OpenCV not found, so we won't build the webcam_face_pose_ex example.")
   endif()
endif()



#here we apply our macros
