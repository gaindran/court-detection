# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/gaindran/demo/court-detection

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/gaindran/demo/court-detection

# Include any dependencies generated for this target.
include CMakeFiles/detect.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/detect.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/detect.dir/flags.make

CMakeFiles/detect.dir/main.cpp.o: CMakeFiles/detect.dir/flags.make
CMakeFiles/detect.dir/main.cpp.o: main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/gaindran/demo/court-detection/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/detect.dir/main.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/detect.dir/main.cpp.o -c /home/gaindran/demo/court-detection/main.cpp

CMakeFiles/detect.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/detect.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/gaindran/demo/court-detection/main.cpp > CMakeFiles/detect.dir/main.cpp.i

CMakeFiles/detect.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/detect.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/gaindran/demo/court-detection/main.cpp -o CMakeFiles/detect.dir/main.cpp.s

CMakeFiles/detect.dir/main.cpp.o.requires:

.PHONY : CMakeFiles/detect.dir/main.cpp.o.requires

CMakeFiles/detect.dir/main.cpp.o.provides: CMakeFiles/detect.dir/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/detect.dir/build.make CMakeFiles/detect.dir/main.cpp.o.provides.build
.PHONY : CMakeFiles/detect.dir/main.cpp.o.provides

CMakeFiles/detect.dir/main.cpp.o.provides.build: CMakeFiles/detect.dir/main.cpp.o


CMakeFiles/detect.dir/CourtLineCandidateDetector.cpp.o: CMakeFiles/detect.dir/flags.make
CMakeFiles/detect.dir/CourtLineCandidateDetector.cpp.o: CourtLineCandidateDetector.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/gaindran/demo/court-detection/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/detect.dir/CourtLineCandidateDetector.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/detect.dir/CourtLineCandidateDetector.cpp.o -c /home/gaindran/demo/court-detection/CourtLineCandidateDetector.cpp

CMakeFiles/detect.dir/CourtLineCandidateDetector.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/detect.dir/CourtLineCandidateDetector.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/gaindran/demo/court-detection/CourtLineCandidateDetector.cpp > CMakeFiles/detect.dir/CourtLineCandidateDetector.cpp.i

CMakeFiles/detect.dir/CourtLineCandidateDetector.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/detect.dir/CourtLineCandidateDetector.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/gaindran/demo/court-detection/CourtLineCandidateDetector.cpp -o CMakeFiles/detect.dir/CourtLineCandidateDetector.cpp.s

CMakeFiles/detect.dir/CourtLineCandidateDetector.cpp.o.requires:

.PHONY : CMakeFiles/detect.dir/CourtLineCandidateDetector.cpp.o.requires

CMakeFiles/detect.dir/CourtLineCandidateDetector.cpp.o.provides: CMakeFiles/detect.dir/CourtLineCandidateDetector.cpp.o.requires
	$(MAKE) -f CMakeFiles/detect.dir/build.make CMakeFiles/detect.dir/CourtLineCandidateDetector.cpp.o.provides.build
.PHONY : CMakeFiles/detect.dir/CourtLineCandidateDetector.cpp.o.provides

CMakeFiles/detect.dir/CourtLineCandidateDetector.cpp.o.provides.build: CMakeFiles/detect.dir/CourtLineCandidateDetector.cpp.o


CMakeFiles/detect.dir/CourtLinePixelDetector.cpp.o: CMakeFiles/detect.dir/flags.make
CMakeFiles/detect.dir/CourtLinePixelDetector.cpp.o: CourtLinePixelDetector.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/gaindran/demo/court-detection/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/detect.dir/CourtLinePixelDetector.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/detect.dir/CourtLinePixelDetector.cpp.o -c /home/gaindran/demo/court-detection/CourtLinePixelDetector.cpp

CMakeFiles/detect.dir/CourtLinePixelDetector.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/detect.dir/CourtLinePixelDetector.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/gaindran/demo/court-detection/CourtLinePixelDetector.cpp > CMakeFiles/detect.dir/CourtLinePixelDetector.cpp.i

CMakeFiles/detect.dir/CourtLinePixelDetector.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/detect.dir/CourtLinePixelDetector.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/gaindran/demo/court-detection/CourtLinePixelDetector.cpp -o CMakeFiles/detect.dir/CourtLinePixelDetector.cpp.s

CMakeFiles/detect.dir/CourtLinePixelDetector.cpp.o.requires:

.PHONY : CMakeFiles/detect.dir/CourtLinePixelDetector.cpp.o.requires

CMakeFiles/detect.dir/CourtLinePixelDetector.cpp.o.provides: CMakeFiles/detect.dir/CourtLinePixelDetector.cpp.o.requires
	$(MAKE) -f CMakeFiles/detect.dir/build.make CMakeFiles/detect.dir/CourtLinePixelDetector.cpp.o.provides.build
.PHONY : CMakeFiles/detect.dir/CourtLinePixelDetector.cpp.o.provides

CMakeFiles/detect.dir/CourtLinePixelDetector.cpp.o.provides.build: CMakeFiles/detect.dir/CourtLinePixelDetector.cpp.o


CMakeFiles/detect.dir/DebugHelpers.cpp.o: CMakeFiles/detect.dir/flags.make
CMakeFiles/detect.dir/DebugHelpers.cpp.o: DebugHelpers.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/gaindran/demo/court-detection/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/detect.dir/DebugHelpers.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/detect.dir/DebugHelpers.cpp.o -c /home/gaindran/demo/court-detection/DebugHelpers.cpp

CMakeFiles/detect.dir/DebugHelpers.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/detect.dir/DebugHelpers.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/gaindran/demo/court-detection/DebugHelpers.cpp > CMakeFiles/detect.dir/DebugHelpers.cpp.i

CMakeFiles/detect.dir/DebugHelpers.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/detect.dir/DebugHelpers.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/gaindran/demo/court-detection/DebugHelpers.cpp -o CMakeFiles/detect.dir/DebugHelpers.cpp.s

CMakeFiles/detect.dir/DebugHelpers.cpp.o.requires:

.PHONY : CMakeFiles/detect.dir/DebugHelpers.cpp.o.requires

CMakeFiles/detect.dir/DebugHelpers.cpp.o.provides: CMakeFiles/detect.dir/DebugHelpers.cpp.o.requires
	$(MAKE) -f CMakeFiles/detect.dir/build.make CMakeFiles/detect.dir/DebugHelpers.cpp.o.provides.build
.PHONY : CMakeFiles/detect.dir/DebugHelpers.cpp.o.provides

CMakeFiles/detect.dir/DebugHelpers.cpp.o.provides.build: CMakeFiles/detect.dir/DebugHelpers.cpp.o


CMakeFiles/detect.dir/geometry.cpp.o: CMakeFiles/detect.dir/flags.make
CMakeFiles/detect.dir/geometry.cpp.o: geometry.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/gaindran/demo/court-detection/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/detect.dir/geometry.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/detect.dir/geometry.cpp.o -c /home/gaindran/demo/court-detection/geometry.cpp

CMakeFiles/detect.dir/geometry.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/detect.dir/geometry.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/gaindran/demo/court-detection/geometry.cpp > CMakeFiles/detect.dir/geometry.cpp.i

CMakeFiles/detect.dir/geometry.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/detect.dir/geometry.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/gaindran/demo/court-detection/geometry.cpp -o CMakeFiles/detect.dir/geometry.cpp.s

CMakeFiles/detect.dir/geometry.cpp.o.requires:

.PHONY : CMakeFiles/detect.dir/geometry.cpp.o.requires

CMakeFiles/detect.dir/geometry.cpp.o.provides: CMakeFiles/detect.dir/geometry.cpp.o.requires
	$(MAKE) -f CMakeFiles/detect.dir/build.make CMakeFiles/detect.dir/geometry.cpp.o.provides.build
.PHONY : CMakeFiles/detect.dir/geometry.cpp.o.provides

CMakeFiles/detect.dir/geometry.cpp.o.provides.build: CMakeFiles/detect.dir/geometry.cpp.o


CMakeFiles/detect.dir/GlobalParameters.cpp.o: CMakeFiles/detect.dir/flags.make
CMakeFiles/detect.dir/GlobalParameters.cpp.o: GlobalParameters.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/gaindran/demo/court-detection/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/detect.dir/GlobalParameters.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/detect.dir/GlobalParameters.cpp.o -c /home/gaindran/demo/court-detection/GlobalParameters.cpp

CMakeFiles/detect.dir/GlobalParameters.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/detect.dir/GlobalParameters.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/gaindran/demo/court-detection/GlobalParameters.cpp > CMakeFiles/detect.dir/GlobalParameters.cpp.i

CMakeFiles/detect.dir/GlobalParameters.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/detect.dir/GlobalParameters.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/gaindran/demo/court-detection/GlobalParameters.cpp -o CMakeFiles/detect.dir/GlobalParameters.cpp.s

CMakeFiles/detect.dir/GlobalParameters.cpp.o.requires:

.PHONY : CMakeFiles/detect.dir/GlobalParameters.cpp.o.requires

CMakeFiles/detect.dir/GlobalParameters.cpp.o.provides: CMakeFiles/detect.dir/GlobalParameters.cpp.o.requires
	$(MAKE) -f CMakeFiles/detect.dir/build.make CMakeFiles/detect.dir/GlobalParameters.cpp.o.provides.build
.PHONY : CMakeFiles/detect.dir/GlobalParameters.cpp.o.provides

CMakeFiles/detect.dir/GlobalParameters.cpp.o.provides.build: CMakeFiles/detect.dir/GlobalParameters.cpp.o


CMakeFiles/detect.dir/LicenseChecker.cpp.o: CMakeFiles/detect.dir/flags.make
CMakeFiles/detect.dir/LicenseChecker.cpp.o: LicenseChecker.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/gaindran/demo/court-detection/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object CMakeFiles/detect.dir/LicenseChecker.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/detect.dir/LicenseChecker.cpp.o -c /home/gaindran/demo/court-detection/LicenseChecker.cpp

CMakeFiles/detect.dir/LicenseChecker.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/detect.dir/LicenseChecker.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/gaindran/demo/court-detection/LicenseChecker.cpp > CMakeFiles/detect.dir/LicenseChecker.cpp.i

CMakeFiles/detect.dir/LicenseChecker.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/detect.dir/LicenseChecker.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/gaindran/demo/court-detection/LicenseChecker.cpp -o CMakeFiles/detect.dir/LicenseChecker.cpp.s

CMakeFiles/detect.dir/LicenseChecker.cpp.o.requires:

.PHONY : CMakeFiles/detect.dir/LicenseChecker.cpp.o.requires

CMakeFiles/detect.dir/LicenseChecker.cpp.o.provides: CMakeFiles/detect.dir/LicenseChecker.cpp.o.requires
	$(MAKE) -f CMakeFiles/detect.dir/build.make CMakeFiles/detect.dir/LicenseChecker.cpp.o.provides.build
.PHONY : CMakeFiles/detect.dir/LicenseChecker.cpp.o.provides

CMakeFiles/detect.dir/LicenseChecker.cpp.o.provides.build: CMakeFiles/detect.dir/LicenseChecker.cpp.o


CMakeFiles/detect.dir/Line.cpp.o: CMakeFiles/detect.dir/flags.make
CMakeFiles/detect.dir/Line.cpp.o: Line.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/gaindran/demo/court-detection/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object CMakeFiles/detect.dir/Line.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/detect.dir/Line.cpp.o -c /home/gaindran/demo/court-detection/Line.cpp

CMakeFiles/detect.dir/Line.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/detect.dir/Line.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/gaindran/demo/court-detection/Line.cpp > CMakeFiles/detect.dir/Line.cpp.i

CMakeFiles/detect.dir/Line.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/detect.dir/Line.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/gaindran/demo/court-detection/Line.cpp -o CMakeFiles/detect.dir/Line.cpp.s

CMakeFiles/detect.dir/Line.cpp.o.requires:

.PHONY : CMakeFiles/detect.dir/Line.cpp.o.requires

CMakeFiles/detect.dir/Line.cpp.o.provides: CMakeFiles/detect.dir/Line.cpp.o.requires
	$(MAKE) -f CMakeFiles/detect.dir/build.make CMakeFiles/detect.dir/Line.cpp.o.provides.build
.PHONY : CMakeFiles/detect.dir/Line.cpp.o.provides

CMakeFiles/detect.dir/Line.cpp.o.provides.build: CMakeFiles/detect.dir/Line.cpp.o


CMakeFiles/detect.dir/TennisCourtFitter.cpp.o: CMakeFiles/detect.dir/flags.make
CMakeFiles/detect.dir/TennisCourtFitter.cpp.o: TennisCourtFitter.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/gaindran/demo/court-detection/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CXX object CMakeFiles/detect.dir/TennisCourtFitter.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/detect.dir/TennisCourtFitter.cpp.o -c /home/gaindran/demo/court-detection/TennisCourtFitter.cpp

CMakeFiles/detect.dir/TennisCourtFitter.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/detect.dir/TennisCourtFitter.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/gaindran/demo/court-detection/TennisCourtFitter.cpp > CMakeFiles/detect.dir/TennisCourtFitter.cpp.i

CMakeFiles/detect.dir/TennisCourtFitter.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/detect.dir/TennisCourtFitter.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/gaindran/demo/court-detection/TennisCourtFitter.cpp -o CMakeFiles/detect.dir/TennisCourtFitter.cpp.s

CMakeFiles/detect.dir/TennisCourtFitter.cpp.o.requires:

.PHONY : CMakeFiles/detect.dir/TennisCourtFitter.cpp.o.requires

CMakeFiles/detect.dir/TennisCourtFitter.cpp.o.provides: CMakeFiles/detect.dir/TennisCourtFitter.cpp.o.requires
	$(MAKE) -f CMakeFiles/detect.dir/build.make CMakeFiles/detect.dir/TennisCourtFitter.cpp.o.provides.build
.PHONY : CMakeFiles/detect.dir/TennisCourtFitter.cpp.o.provides

CMakeFiles/detect.dir/TennisCourtFitter.cpp.o.provides.build: CMakeFiles/detect.dir/TennisCourtFitter.cpp.o


CMakeFiles/detect.dir/TennisCourtModel.cpp.o: CMakeFiles/detect.dir/flags.make
CMakeFiles/detect.dir/TennisCourtModel.cpp.o: TennisCourtModel.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/gaindran/demo/court-detection/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Building CXX object CMakeFiles/detect.dir/TennisCourtModel.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/detect.dir/TennisCourtModel.cpp.o -c /home/gaindran/demo/court-detection/TennisCourtModel.cpp

CMakeFiles/detect.dir/TennisCourtModel.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/detect.dir/TennisCourtModel.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/gaindran/demo/court-detection/TennisCourtModel.cpp > CMakeFiles/detect.dir/TennisCourtModel.cpp.i

CMakeFiles/detect.dir/TennisCourtModel.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/detect.dir/TennisCourtModel.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/gaindran/demo/court-detection/TennisCourtModel.cpp -o CMakeFiles/detect.dir/TennisCourtModel.cpp.s

CMakeFiles/detect.dir/TennisCourtModel.cpp.o.requires:

.PHONY : CMakeFiles/detect.dir/TennisCourtModel.cpp.o.requires

CMakeFiles/detect.dir/TennisCourtModel.cpp.o.provides: CMakeFiles/detect.dir/TennisCourtModel.cpp.o.requires
	$(MAKE) -f CMakeFiles/detect.dir/build.make CMakeFiles/detect.dir/TennisCourtModel.cpp.o.provides.build
.PHONY : CMakeFiles/detect.dir/TennisCourtModel.cpp.o.provides

CMakeFiles/detect.dir/TennisCourtModel.cpp.o.provides.build: CMakeFiles/detect.dir/TennisCourtModel.cpp.o


CMakeFiles/detect.dir/TimeMeasurement.cpp.o: CMakeFiles/detect.dir/flags.make
CMakeFiles/detect.dir/TimeMeasurement.cpp.o: TimeMeasurement.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/gaindran/demo/court-detection/CMakeFiles --progress-num=$(CMAKE_PROGRESS_11) "Building CXX object CMakeFiles/detect.dir/TimeMeasurement.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/detect.dir/TimeMeasurement.cpp.o -c /home/gaindran/demo/court-detection/TimeMeasurement.cpp

CMakeFiles/detect.dir/TimeMeasurement.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/detect.dir/TimeMeasurement.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/gaindran/demo/court-detection/TimeMeasurement.cpp > CMakeFiles/detect.dir/TimeMeasurement.cpp.i

CMakeFiles/detect.dir/TimeMeasurement.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/detect.dir/TimeMeasurement.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/gaindran/demo/court-detection/TimeMeasurement.cpp -o CMakeFiles/detect.dir/TimeMeasurement.cpp.s

CMakeFiles/detect.dir/TimeMeasurement.cpp.o.requires:

.PHONY : CMakeFiles/detect.dir/TimeMeasurement.cpp.o.requires

CMakeFiles/detect.dir/TimeMeasurement.cpp.o.provides: CMakeFiles/detect.dir/TimeMeasurement.cpp.o.requires
	$(MAKE) -f CMakeFiles/detect.dir/build.make CMakeFiles/detect.dir/TimeMeasurement.cpp.o.provides.build
.PHONY : CMakeFiles/detect.dir/TimeMeasurement.cpp.o.provides

CMakeFiles/detect.dir/TimeMeasurement.cpp.o.provides.build: CMakeFiles/detect.dir/TimeMeasurement.cpp.o


# Object files for target detect
detect_OBJECTS = \
"CMakeFiles/detect.dir/main.cpp.o" \
"CMakeFiles/detect.dir/CourtLineCandidateDetector.cpp.o" \
"CMakeFiles/detect.dir/CourtLinePixelDetector.cpp.o" \
"CMakeFiles/detect.dir/DebugHelpers.cpp.o" \
"CMakeFiles/detect.dir/geometry.cpp.o" \
"CMakeFiles/detect.dir/GlobalParameters.cpp.o" \
"CMakeFiles/detect.dir/LicenseChecker.cpp.o" \
"CMakeFiles/detect.dir/Line.cpp.o" \
"CMakeFiles/detect.dir/TennisCourtFitter.cpp.o" \
"CMakeFiles/detect.dir/TennisCourtModel.cpp.o" \
"CMakeFiles/detect.dir/TimeMeasurement.cpp.o"

# External object files for target detect
detect_EXTERNAL_OBJECTS =

detect: CMakeFiles/detect.dir/main.cpp.o
detect: CMakeFiles/detect.dir/CourtLineCandidateDetector.cpp.o
detect: CMakeFiles/detect.dir/CourtLinePixelDetector.cpp.o
detect: CMakeFiles/detect.dir/DebugHelpers.cpp.o
detect: CMakeFiles/detect.dir/geometry.cpp.o
detect: CMakeFiles/detect.dir/GlobalParameters.cpp.o
detect: CMakeFiles/detect.dir/LicenseChecker.cpp.o
detect: CMakeFiles/detect.dir/Line.cpp.o
detect: CMakeFiles/detect.dir/TennisCourtFitter.cpp.o
detect: CMakeFiles/detect.dir/TennisCourtModel.cpp.o
detect: CMakeFiles/detect.dir/TimeMeasurement.cpp.o
detect: CMakeFiles/detect.dir/build.make
detect: /usr/local/lib/libopencv_dnn.so.3.4.8
detect: /usr/local/lib/libopencv_highgui.so.3.4.8
detect: /usr/local/lib/libopencv_ml.so.3.4.8
detect: /usr/local/lib/libopencv_objdetect.so.3.4.8
detect: /usr/local/lib/libopencv_shape.so.3.4.8
detect: /usr/local/lib/libopencv_stitching.so.3.4.8
detect: /usr/local/lib/libopencv_superres.so.3.4.8
detect: /usr/local/lib/libopencv_videostab.so.3.4.8
detect: /usr/local/lib/libopencv_calib3d.so.3.4.8
detect: /usr/local/lib/libopencv_features2d.so.3.4.8
detect: /usr/local/lib/libopencv_flann.so.3.4.8
detect: /usr/local/lib/libopencv_photo.so.3.4.8
detect: /usr/local/lib/libopencv_video.so.3.4.8
detect: /usr/local/lib/libopencv_videoio.so.3.4.8
detect: /usr/local/lib/libopencv_imgcodecs.so.3.4.8
detect: /usr/local/lib/libopencv_imgproc.so.3.4.8
detect: /usr/local/lib/libopencv_core.so.3.4.8
detect: CMakeFiles/detect.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/gaindran/demo/court-detection/CMakeFiles --progress-num=$(CMAKE_PROGRESS_12) "Linking CXX executable detect"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/detect.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/detect.dir/build: detect

.PHONY : CMakeFiles/detect.dir/build

CMakeFiles/detect.dir/requires: CMakeFiles/detect.dir/main.cpp.o.requires
CMakeFiles/detect.dir/requires: CMakeFiles/detect.dir/CourtLineCandidateDetector.cpp.o.requires
CMakeFiles/detect.dir/requires: CMakeFiles/detect.dir/CourtLinePixelDetector.cpp.o.requires
CMakeFiles/detect.dir/requires: CMakeFiles/detect.dir/DebugHelpers.cpp.o.requires
CMakeFiles/detect.dir/requires: CMakeFiles/detect.dir/geometry.cpp.o.requires
CMakeFiles/detect.dir/requires: CMakeFiles/detect.dir/GlobalParameters.cpp.o.requires
CMakeFiles/detect.dir/requires: CMakeFiles/detect.dir/LicenseChecker.cpp.o.requires
CMakeFiles/detect.dir/requires: CMakeFiles/detect.dir/Line.cpp.o.requires
CMakeFiles/detect.dir/requires: CMakeFiles/detect.dir/TennisCourtFitter.cpp.o.requires
CMakeFiles/detect.dir/requires: CMakeFiles/detect.dir/TennisCourtModel.cpp.o.requires
CMakeFiles/detect.dir/requires: CMakeFiles/detect.dir/TimeMeasurement.cpp.o.requires

.PHONY : CMakeFiles/detect.dir/requires

CMakeFiles/detect.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/detect.dir/cmake_clean.cmake
.PHONY : CMakeFiles/detect.dir/clean

CMakeFiles/detect.dir/depend:
	cd /home/gaindran/demo/court-detection && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/gaindran/demo/court-detection /home/gaindran/demo/court-detection /home/gaindran/demo/court-detection /home/gaindran/demo/court-detection /home/gaindran/demo/court-detection/CMakeFiles/detect.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/detect.dir/depend

