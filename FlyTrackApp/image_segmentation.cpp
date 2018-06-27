#include "image_segmentation.h"

#include "frames.h"

#include <string>
#include <stdint.h>
#include <tuple>
#include <iostream>
#include <cmath>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/ximgproc/segmentation.hpp"
#include <opencv2/video/background_segm.hpp>
#include <armadillo>

using namespace cv::ximgproc::segmentation;
using namespace std;

ImagSegm::ImagSegm() {
	// empty
}

void ImagSegm::SetBodySegmentationParam(int BodyThresh, int BodyBlurSigma, int BodyBlurWindow, double BodySigma, int BodyK, int MinBodySize, double BodyLength, vector<tuple<double,double>> Origin, bool Tethered) {

	body_thresh = BodyThresh;
	body_blur_sigma = BodyBlurSigma;
	body_blur_window = BodyBlurWindow;
	body_sigma = BodySigma;
	body_K = BodyK;
	min_body_size = MinBodySize;
	body_length = BodyLength;
	origin = Origin;
	tethered = Tethered;

}

void ImagSegm::SetWingSegmentationParam(int WingThresh, int BodyDilation, double WingSigma, int WingK, int MinWingSize, double WingLength) {

	wing_thresh = WingThresh;
	body_dilation = BodyDilation;
	wing_sigma = WingSigma;
	wing_K = WingK;
	min_wing_size = MinWingSize;
	wing_length = WingLength;

}

void ImagSegm::SegmentSingleFrame(frames &frame_in) {

	int N_cam = frame_in.bckg_images.size();

	frame_in.single_seg_frame.clear();

	for (int n=0; n<N_cam; n++) {

		arma::Col<int> img_vec = frame_in.single_raw_frame[n];
		int N_row = get<0>(frame_in.image_size[n]);
		int N_col = get<1>(frame_in.image_size[n]);

		// Get body segment:
		arma::Col<int> body_thresh_img = ImagSegm::BodyThresh(img_vec, N_row, N_col, body_thresh, body_blur_sigma, body_blur_window);
		arma::Col<int> body_seg_img = ImagSegm::GraphSeg(body_thresh_img, N_row, N_col, body_sigma, body_K, min_body_size);
		arma::Mat<double> body_seg_prop = ImagSegm::GetSegmentProperties(body_seg_img, N_row, N_col);
		tuple<arma::Col<double>,arma::Col<int>> body_seg = ImagSegm::SelectBody(body_seg_img, N_row, N_col, body_seg_prop, origin[n]);

		// Get wing segments:
		arma::Col<int> wing_thresh_img = ImagSegm::WingThresh(img_vec, get<1>(body_seg), N_row, N_col, wing_thresh, body_dilation, get<0>(body_seg));
		arma::Col<int> wing_seg_img = ImagSegm::GraphSeg(wing_thresh_img, N_row, N_col, wing_sigma, wing_K, min_wing_size);
		arma::Mat<double> wing_seg_prop = ImagSegm::GetSegmentProperties(wing_seg_img, N_row, N_col);
		tuple<arma::Mat<double>,arma::Mat<int>> wing_seg = ImagSegm::SelectWing(wing_seg_img, N_row, N_col, wing_seg_prop, get<0>(body_seg));

		// Construct segmented frame:
		arma::Col<int> body_frame = get<1>(body_seg);
		arma::Col<double> body_frame_d = arma::conv_to<arma::Mat<double>>::from(body_frame);
		arma::Mat<int> wing_frame = get<1>(wing_seg);
		arma::Mat<double> wing_frame_d = arma::conv_to<arma::Mat<double>>::from(wing_frame);

		arma::Col<double> temp_frame(N_row*N_col);
		temp_frame.zeros();

		for (int j=0; j<(wing_frame.n_cols+1); j++) {
			if (j==0) {
				temp_frame += body_frame_d*(1.0/255.0);
			}
			else {
				temp_frame += wing_frame_d.col(j-1)*((j+1)/255.0);
			}
		}

		arma::Col<int> seg_frame = arma::conv_to<arma::Mat<int>>::from(temp_frame);

		// Load segmented frame into the frames struct:
		frame_in.single_seg_frame.push_back(seg_frame);

	}

}

void ImagSegm::SegmentFrameBatch(frames &frame_in) {

	int N_cam = frame_in.bckg_images.size();

	int N_batch = frame_in.raw_frames[0].n_cols;

	frame_in.seg_frames.clear();

	for (int n=0; n<N_cam; n++) {

		int N_row = get<0>(frame_in.image_size[n]);
		int N_col = get<1>(frame_in.image_size[n]);

		arma::Mat<int> seg_frame_mat(N_row*N_col,N_batch);

		for (int i=0; i<N_batch; i++) {

			arma::Col<int> img_vec = frame_in.raw_frames[n].col(i);

			// Get body segment:
			arma::Col<int> body_thresh_img = ImagSegm::BodyThresh(img_vec, N_row, N_col, body_thresh, body_blur_sigma, body_blur_window);
			arma::Col<int> body_seg_img = ImagSegm::GraphSeg(body_thresh_img, N_row, N_col, body_sigma, body_K, min_body_size);
			arma::Mat<double> body_seg_prop = ImagSegm::GetSegmentProperties(body_seg_img, N_row, N_col);
			tuple<arma::Col<double>,arma::Col<int>> body_seg = ImagSegm::SelectBody(body_seg_img, N_row, N_col, body_seg_prop, origin[n]);

			// Get wing segments:
			arma::Col<int> wing_thresh_img = ImagSegm::WingThresh(img_vec, get<1>(body_seg), N_row, N_col, wing_thresh, body_dilation, get<0>(body_seg));
			arma::Col<int> wing_seg_img = ImagSegm::GraphSeg(wing_thresh_img, N_row, N_col, wing_sigma, wing_K, min_wing_size);
			arma::Mat<double> wing_seg_prop = ImagSegm::GetSegmentProperties(wing_seg_img, N_row, N_col);
			tuple<arma::Mat<double>,arma::Mat<int>> wing_seg = ImagSegm::SelectWing(wing_seg_img, N_row, N_col, wing_seg_prop, get<0>(body_seg));

			// Construct segmented frame:
			arma::Col<int> body_frame = get<1>(body_seg);
			arma::Col<double> body_frame_d = arma::conv_to<arma::Mat<double>>::from(body_frame);
			arma::Mat<int> wing_frame = get<1>(wing_seg);
			arma::Mat<double> wing_frame_d = arma::conv_to<arma::Mat<double>>::from(wing_frame);

			arma::Col<double> temp_frame(N_row*N_col);
			temp_frame.zeros();

			for (int j=0; j<(wing_frame.n_cols+1); j++) {
				if (j==0) {
					temp_frame += body_frame_d*(1.0/255.0);
				}
				else {
					temp_frame += wing_frame_d.col(j-1)*((j+1)/255.0);
				}
			}

			seg_frame_mat.col(i) = arma::conv_to<arma::Mat<int>>::from(temp_frame);

		}

		// Load segmented frame into the frames struct:
		frame_in.raw_frames.push_back(seg_frame_mat);

	}

}

tuple<arma::Col<double>,arma::Col<int>> ImagSegm::SelectBody(arma::Col<int> &frame_in, int N_row, int N_col, arma::Mat<double> seg_prop, tuple<double,double> origin_loc) {

	int N_seg = frame_in.max()+1;
	bool is_body = true;
	tuple<arma::Col<double>, arma::Col<int>> body_img;

	vector<tuple<int,double>> pos_body_segs;

	if (N_seg == 1) {
		cout << "no body segment found" << endl;
		arma::Col<int> body_img_vec(N_row*N_col);
		body_img_vec.zeros();
		arma::Col<double> body_prop_vec(4);
		body_prop_vec.ones();
		body_img = make_tuple((body_prop_vec*-1.0),body_img_vec);
	}
	else {
		for (int i=0; i<N_seg; i++) {
			is_body = true;

			if (tethered==false) {
				// Body origin location is unknown (free flight):
				if (seg_prop(2,i)<(0.1*pow(body_length,2))) {
					is_body = false;
				}
				if (seg_prop(2,i)>(0.75*pow(body_length,2))) {
					is_body = false;
				}
				if (seg_prop(3,i)>0) {
					is_body = false;
				}
			}
			else {
				// Body origin location is known (tethered flight):
				double dist_origin = sqrt(pow(seg_prop(0,i)-get<0>(origin_loc),2)+pow(seg_prop(1,i)-get<1>(origin_loc),2));
				if (dist_origin>(0.75*body_length)) {
					is_body = false;
				}
				if (seg_prop(2,i)<(0.1*pow(body_length,2))) {
					is_body = false;
				}
				if (seg_prop(2,i)>(0.75*pow(body_length,2))) {
					is_body = false;
				}
				if (seg_prop(3,i)>0) {
					is_body = false;
				}
			}

			if (is_body == true) {
				pos_body_segs.push_back(make_tuple(i,seg_prop(2,i)));
			}
		}

		if (pos_body_segs.size()==1) {
			// Only one candidate for the body segment.
			cv::Mat seg_img, bin_img;
			seg_img = ImagSegm::Vector2CVMat(frame_in,N_row,N_col);
			cv::inRange(seg_img,get<0>(pos_body_segs[0]),get<0>(pos_body_segs[0]),bin_img);
			arma::Col<int> body_img_vec = ImagSegm::CVMat2Vector(bin_img);
			body_img = make_tuple(seg_prop.col(get<0>(pos_body_segs[0])),body_img_vec);
		}
		else if (pos_body_segs.size()>1) {
			// More than one candidate for the body segment, select the segment that is closest to 0.25*pow(body_length,2)
			arma::Col<double> area_diff(pos_body_segs.size());
			for (int j=0; j<pos_body_segs.size(); j++) {
				area_diff(j) = abs(get<1>(pos_body_segs[j])-0.25*pow(body_length,2));
			}
			int min_ind = area_diff.index_min();
			cv::Mat seg_img, bin_img;
			seg_img = ImagSegm::Vector2CVMat(frame_in,N_row,N_col);
			cv::inRange(seg_img,get<0>(pos_body_segs[min_ind]),get<0>(pos_body_segs[min_ind]),bin_img);
			arma::Col<int> body_img_vec = ImagSegm::CVMat2Vector(bin_img);
			body_img = make_tuple(seg_prop.col(min_ind),body_img_vec);
		}
		else {
			// No candidate for the body segment
			arma::Col<int> body_img_vec(N_row*N_col);
			body_img_vec.zeros();
			arma::Col<double> body_prop_vec(4);
			body_prop_vec.ones();
			body_img = make_tuple((body_prop_vec*-1.0),body_img_vec);
		}
	}

	return body_img;	
}

tuple<arma::Mat<double>,arma::Mat<int>> ImagSegm::SelectWing(arma::Col<int> &frame_in, int N_row, int N_col, arma::Mat<double> seg_prop, arma::Col<double> body_prop) {

	int N_seg = frame_in.max()+1;
	bool is_wing = true;
	tuple<arma::Mat<double>,arma::Mat<int>> wing_img;

	vector<int> wing_segs;

	if (N_seg == 1) {
		cout << "no wing segment found" << endl;
		arma::Mat<int> wing_img_vec(N_row*N_col,1);
		wing_img_vec.zeros();
		arma::Mat<double> wing_prop_vec(4,1);
		wing_prop_vec.ones();
		wing_img = make_tuple((wing_prop_vec*-1.0),wing_img_vec);
	}
	else {
		for (int i=0; i<N_seg; i++) {
			is_wing = true;

			double dist_cg = sqrt(pow(seg_prop(0,i)-body_prop(0),2)+pow(seg_prop(1,i)-body_prop(1),2));

			if (dist_cg>(1.5*wing_length)) {
				is_wing = false;
			}
			if (seg_prop(2,i)>(2.0*pow(wing_length,2))) {
				is_wing = false;
			}
			if (seg_prop(3,i)>0) {
				is_wing = false;
			}
			
			if (is_wing == true) {
				wing_segs.push_back(i);
			}
		}

		if (wing_segs.size()>0) {
			arma::Mat<int> wing_img_vec(N_row*N_col,wing_segs.size());
			arma::Mat<double> wing_prop_vec(4,wing_segs.size());
			for (int j=0; j<wing_segs.size(); j++) {
				cv::Mat seg_img, bin_img;
				seg_img = ImagSegm::Vector2CVMat(frame_in,N_row,N_col);
				cv::inRange(seg_img,wing_segs[j],wing_segs[j],bin_img);
				wing_img_vec.col(j) = ImagSegm::CVMat2Vector(bin_img);
				wing_prop_vec.col(j) = seg_prop.col(wing_segs[j]);
			}
			wing_img = make_tuple(wing_prop_vec,wing_img_vec);
		}
		else {
			cout << "no wing segment found" << endl;
			arma::Mat<int> wing_img_vec(N_row*N_col,1);
			wing_img_vec.zeros();
			arma::Mat<double> wing_prop_vec(4,1);
			wing_prop_vec.ones();
			wing_img = make_tuple((wing_prop_vec*-1.0),wing_img_vec);
		}
	}

	return wing_img;
}

arma::Col<int> ImagSegm::BodyThresh(arma::Col<int> &frame_in, int N_row, int N_col, int body_thresh, int sigma_blur, int blur_window) {

	cv::Mat img, ThreshImg;

	img = ImagSegm::Vector2CVMat(frame_in,N_row,N_col);

	cv::GaussianBlur(img, img, cv::Size(blur_window,blur_window),sigma_blur,sigma_blur);

	cv::threshold(img, ThreshImg, body_thresh, 0, cv::THRESH_TOZERO_INV);

	arma::Col<int> thresh_vec; 

	thresh_vec = ImagSegm::CVMat2Vector(ThreshImg);

	return thresh_vec;
}

arma::Col<int> ImagSegm::WingThresh(arma::Col<int> &frame_in, arma::Col<int> &body_frame_in, int N_row, int N_col, int wing_thresh, int body_dilation, arma::Col<double> body_prop) {

	cv::Mat img, img_coords_body, body_img, circle_mask, ThreshImg;

	img = ImagSegm::Vector2CVMat(frame_in,N_row,N_col);

	body_img = cv::Mat::zeros(N_row, N_col, CV_8UC1 );

	body_img = body_img+ImagSegm::Vector2CVMat(body_frame_in,N_row,N_col);

	// Dilate the image a bit:
	if (body_dilation > 0) {
		cv::dilate(body_img, body_img, cv::Mat(), cv::Point(-1, -1), body_dilation, 1, 1);
	}

	// Create a circular mask of 1.6 * wing length around the body center
	//double radius = 1.6*wing_length;
	//circle_mask = cv::Mat::zeros(N_row, N_col, CV_8UC1 );
	//cv::circle(circle_mask,cv::Point(body_prop(0),body_prop(1)),radius,cv::Scalar::all(255),-1);

	// Perform thresholding:
	cv::threshold(255-(img+body_img), ThreshImg, wing_thresh, 255, cv::THRESH_TOZERO);
	//cv::threshold(255-(img+body_img)-(255-circle_mask), ThreshImg, wing_thresh, 255, cv::THRESH_TOZERO);

	arma::Col<int> thresh_vec;

	thresh_vec = ImagSegm::CVMat2Vector(ThreshImg);

	return thresh_vec;
}

arma::Mat<double> ImagSegm::GetSegmentProperties(arma::Col<int> &frame_in, int N_row, int N_col) {

	int N_seg = frame_in.max()+1;

	arma::Mat<double> seg_prop(4,N_seg);

	cv::Mat seg_img = ImagSegm::Vector2CVMat(frame_in, N_row, N_col);

	for (int i=0; i<N_seg; i++) {

		cv::Mat bin_img;

		cv::inRange(seg_img,i,i,bin_img);

		double minVal;
		double maxVal;
		cv::Point minLoc;
		cv::Point maxLoc;

		cv::minMaxLoc(bin_img, &minVal, &maxVal, &minLoc, &maxLoc);

		double cg_x, cg_y, Area, Mx, My, border_sum;

		// Calculate area

		Area = cv::sum(bin_img)[0]/maxVal;

		Mx = 0.0;
		for (int j=0; j<N_row; j++) {
			Mx+=(cv::sum(bin_img.row(j))[0]*j)/maxVal;
		}

		My = 0.0;
		for (int k=0; k<N_col; k++) {
			My+=(cv::sum(bin_img.col(k))[0]*k)/maxVal;
		}

		// Calculate cg

		cg_x = My/Area;
		cg_y = Mx/Area;

		// Check if the segment hits one of the borders
		border_sum = cv::sum(bin_img.row(0))[0]+cv::sum(bin_img.row(N_row-1))[0]+cv::sum(bin_img.col(0))[0]+cv::sum(bin_img.col(N_col-1))[0];

		seg_prop(0,i) = cg_x;
		seg_prop(1,i) = cg_y;
		seg_prop(2,i) = Area;
		seg_prop(3,i) = border_sum;

	}
	return seg_prop;
}

arma::Col<int> ImagSegm::GraphSeg(arma::Col<int> &frame_in, int N_row, int N_col, double Sigma, int K, int minsize) {

	gs->setSigma(Sigma);
	gs->setK(K);
	gs->setMinSize(minsize);

	cv::Mat input_img, output_img;

	input_img = ImagSegm::Vector2CVMat(frame_in,N_row,N_col);

	gs->processImage(input_img,output_img);

	arma::Col<int> segmented_frame;
	segmented_frame = ImagSegm::CVMat2Vector(output_img);

	return segmented_frame;
}

arma::Col<int> ImagSegm::CVMat2Vector(cv::Mat &img) {

	arma::Col<int> img_vec;

	img.convertTo(img, CV_8UC1);
	arma::Mat<uint8_t> img_mat(reinterpret_cast<uint8_t*>(img.data), img.rows, img.cols);
	img_vec = arma::vectorise(arma::conv_to<arma::Mat<int>>::from(img_mat));

	return img_vec;
}

cv::Mat ImagSegm::Vector2CVMat(arma::Col<int> &img_vec, int N_row, int N_col) {

	arma::Mat<int> img_mat;

	img_mat = img_vec;

	img_mat.reshape(N_row,N_col);

	cv::Mat img(img_mat.n_rows, img_mat.n_cols, CV_32SC1, img_mat.memptr());

	img.convertTo(img, CV_8UC1);

	return img;
}

np::ndarray ImagSegm::ReturnSegFrame(session_param &ses_par, frames &frame_now, int cam_nr) {

	int N_row = get<0>(frame_now.image_size[cam_nr]);
	int N_col = get<1>(frame_now.image_size[cam_nr]);

	p::tuple shape = p::make_tuple(N_row,N_col);
	np::dtype dtype = np::dtype::get_builtin<int>();
	np::ndarray array_out = np::zeros(shape,dtype);

	for (int i=0; i<N_row; i++) {
		for (int j=0; j<N_col; j++) {
			array_out[i][j] = frame_now.single_seg_frame[cam_nr](i*N_row+j);
		}
	}

	return array_out;
}