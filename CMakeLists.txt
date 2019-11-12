cmake_minimum_required(VERSION 2.8)
project( court-detect )
find_package( OpenCV REQUIRED )
add_executable( detect main.cpp CourtLineCandidateDetector.cpp CourtLinePixelDetector.cpp DebugHelpers.cpp geometry.cpp GlobalParameters.cpp LicenseChecker.cpp Line.cpp TennisCourtFitter.cpp TennisCourtModel.cpp TimeMeasurement.cpp)
target_link_libraries( detect ${OpenCV_LIBS} )

