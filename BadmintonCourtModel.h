//
// Created by gaindran on 13.11.19.
// Modified from TennisCourtModel.cpp
//
#pragma once

#include "Line.h"

typedef std::pair<Line, Line> LinePair;

class BadmintonCourtModel
{
public:
  BadmintonCourtModel();

  BadmintonCourtModel(const BadmintonCourtModel& o);

  BadmintonCourtModel& operator=(const BadmintonCourtModel& o);

  float fit(const LinePair& hLinePair, const LinePair& vLinePair, const cv::Mat& binaryImage,
    const cv::Mat& rgbImage);

  static std::vector<LinePair> getPossibleLinePairs(std::vector<Line>& lines);

  void drawModel(cv::Mat& image, cv::Scalar color=cv::Scalar(0, 255, 255));

  void writeToFile(const std::string& filename);

private:
  std::vector<cv::Point2f> getIntersectionPoints(const LinePair& hLinePair, const LinePair& vLinePair);

  float evaluateModel(const std::vector<cv::Point2f>& courtPoints, const cv::Mat& binaryImage);

  float computeScoreForLineSegment(const cv::Point2f& start, const cv::Point2f& end,
    const cv::Mat& binaryImage);

  bool isInsideTheImage(float x, float y, const cv::Mat& image);

  void drawModel(std::vector<cv::Point2f>& courtPoints, cv::Mat& image, cv::Scalar color=cv::Scalar(0, 255, 255));

  std::vector<Line> hLines;
  std::vector<Line> vLines;
  std::vector<LinePair> hLinePairs;
  std::vector<LinePair> vLinePairs;
  std::vector<cv::Point2f> courtPoints;
  cv::Mat transformationMatrix;

};