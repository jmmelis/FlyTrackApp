#ifndef FRAME_LOADER_CLASS_H
#define FRAME_LOADER_CLASS_H

#include "session_param.h"
#include "frames.h"

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <string>
#include <stdint.h>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <armadillo>

using namespace std;

namespace p = boost::python;
namespace np = boost::python::numpy;

class FrameLoader {

	public:

		FrameLoader();

		bool LoadBackground(struct session_param &ses_par, struct frames &frame_now);
		bool LoadSingleFrame(struct session_param &ses_par, struct frames &frame_now, int frame_nr);
		bool LoadFrameBatch(struct session_param &ses_par, struct frames &frame_now, int start_frame, int end_frame, int batch_nr);
		arma::Col<int> BackgroundSubtract(arma::Col<int> &img_vec, arma::Col<int> &bckg_vec);
		arma::Col<int> CVMat2Vector(cv::Mat &img);
		cv::Mat Vector2CVMat(arma::Col<int> &img_vec, int N_row, int N_col);
		np::ndarray ReturnSingleFrame(struct session_param &ses_par, struct frames &frame_now, int cam_nr);

};
#endif