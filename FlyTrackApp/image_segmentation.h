#ifndef IMAGE_SEGMENTATION_H
#define IMAGE_SEGMENTATION_H

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
#include "opencv2/ximgproc/segmentation.hpp"
#include <opencv2/video/background_segm.hpp>
#include <armadillo>

using namespace cv::ximgproc::segmentation;
using namespace std;

namespace p = boost::python;
namespace np = boost::python::numpy;

class ImagSegm {

	public:

		ImagSegm();

		int body_thresh;
		int body_blur_sigma;
		int body_blur_window;
		double body_sigma;
		int body_K;
		int min_body_size;
		double body_length;
		vector<tuple<double,double>> origin;
		int wing_thresh;
		int body_dilation;
		double wing_sigma;
		int wing_K;
		int min_wing_size;
		double wing_length;
		bool tethered;

		//cv::setUseOptimized(true);
		//cv::setNumThreads(8);

		cv::Ptr<GraphSegmentation> gs = createGraphSegmentation();

		void SetBodySegmentationParam(int BodyThresh, int BodyBlurSigma, int BodyBlurWindow, double BodySigma, int BodyK, int MinBodySize, double BodyLength, vector<tuple<double,double>> Origin, bool Tethered);
		void SetWingSegmentationParam(int WingThresh, int BodyDilation, double WingSigma, int WingK, int MinWingSize, double WingLength);
		void SegmentSingleFrame(frames &frame_in);
		void SegmentFrameBatch(frames &frame_in);
		tuple<arma::Col<double>,arma::Col<int>> SelectBody(arma::Col<int> &frame_in, int N_row, int N_col, arma::Mat<double> seg_prop, tuple<double,double> origin_loc);
		tuple<arma::Mat<double>,arma::Mat<int>> SelectWing(arma::Col<int> &frame_in, int N_row, int N_col, arma::Mat<double> seg_prop, arma::Col<double> body_prop);
		arma::Col<int> BodyThresh(arma::Col<int> &frame_in, int N_row, int N_col, int body_thresh, int sigma_blur, int blur_window);
		arma::Col<int> WingThresh(arma::Col<int> &frame_in, arma::Col<int> &body_frame_in, int N_row, int N_col, int wing_thresh, int body_dilation, arma::Col<double> body_prop);
		arma::Mat<double> GetSegmentProperties(arma::Col<int> &frame_in, int N_row, int N_col);
		arma::Col<int> GraphSeg(arma::Col<int> &frame_in, int N_row, int N_col, double Sigma, int K, int minsize);
		arma::Col<int> CVMat2Vector(cv::Mat &img);
		cv::Mat Vector2CVMat(arma::Col<int> &img_vec, int N_row, int N_col);
		np::ndarray ReturnSegFrame(session_param &ses_par, frames &frame_now, int cam_nr);

};
#endif