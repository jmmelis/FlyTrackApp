#include "frame_loader.h"

#include <string>
#include <stdint.h>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <armadillo>

using namespace std;

FrameLoader::FrameLoader() {
	// empty
}

bool FrameLoader::LoadBackground(session_param &ses_par, frames &frame_now) {

	bool bckg_loaded = true;

	frame_now.image_size.clear();
	frame_now.bckg_images.clear();

	for (int i=1; i<=ses_par.N_cam; i++) {

		string img_name = ses_par.session_loc+"/"+ses_par.bckg_loc+"/"+ses_par.bckg_name+to_string(i)+"."+ses_par.bckg_img_format;

		cv::Mat image = cv::imread(img_name, CV_LOAD_IMAGE_GRAYSCALE);

		if (image.empty()) {
			bckg_loaded = false;
		}
		else {
			arma::Col<int> bckg_vec = FrameLoader::CVMat2Vector(image);
			frame_now.bckg_images.push_back(bckg_vec);
			tuple<int, int> img_size = make_tuple(image.rows, image.cols);
			frame_now.image_size.push_back(img_size);
		}

	}

	return bckg_loaded;
}

bool FrameLoader::LoadSingleFrame(session_param &ses_par, frames &frame_now, int frame_nr) {

	bool frame_loaded = true;

	frame_now.single_raw_frame.clear();

	for (int i=0; i<ses_par.N_cam; i++) {

		string img_name = ses_par.session_loc+"/"+ses_par.mov_name+to_string(ses_par.mov_nr)+"/"+ses_par.cam_name+to_string(i+1)+"/"+ses_par.frame_name+
						to_string(ses_par.chrono_frame_nr[frame_nr])+"."+ses_par.frame_img_format;

		cv::Mat image = cv::imread(img_name, CV_LOAD_IMAGE_GRAYSCALE);

		if (image.empty()) {
			frame_loaded = false;
		}
		else {
			arma::Col<int> img_vec = FrameLoader::CVMat2Vector(image);
			frame_now.single_raw_frame.push_back(FrameLoader::BackgroundSubtract(img_vec,frame_now.bckg_images[i]));
		}

	}

	return frame_loaded;
}

bool FrameLoader::LoadFrameBatch(session_param &ses_par, frames &frame_now, int start_frame, int end_frame, int batch_nr) {

	bool frames_loaded = true;

	int N_frames = end_frame-start_frame+1;

	frame_now.batch_nr = batch_nr;
	frame_now.start_frame = start_frame;
	frame_now.end_frame = end_frame;
	frame_now.raw_frames.clear();

	for (int i=0; i<ses_par.N_cam; i++) {
		arma::Mat<int> frame_batch(frame_now.bckg_images[i].n_rows,N_frames);
		for (int j=0; j<N_frames; j++) {

			string img_name = ses_par.session_loc+"/"+ses_par.mov_name+to_string(ses_par.mov_nr)+"/"+ses_par.cam_name+to_string(i+1)+"/"+ses_par.frame_name+
							to_string(ses_par.chrono_frame_nr[start_frame+j])+"."+ses_par.frame_img_format;

			cv::Mat image = cv::imread(img_name, CV_LOAD_IMAGE_GRAYSCALE);

			if (image.empty()) {
				frames_loaded = false;
			}
			else {
				arma::Col<int> sub_img = FrameLoader::CVMat2Vector(image);
				frame_batch.col(j) = FrameLoader::BackgroundSubtract(sub_img,frame_now.bckg_images[i]);
			}
		}
		frame_now.raw_frames.push_back(frame_batch);
	}

	return frames_loaded;
}

arma::Col<int> FrameLoader::BackgroundSubtract(arma::Col<int> &img_vec, arma::Col<int> &bckg_vec) {

	arma::Col<int> s255;

	s255.set_size(img_vec.n_rows);

	s255.fill(255);

	arma::Col<int> sub_img = img_vec-(bckg_vec-s255);

	return sub_img;
}

arma::Col<int> FrameLoader::CVMat2Vector(cv::Mat &img) {

	arma::Col<int> img_vec;

	img.convertTo(img, CV_8UC1);
	arma::Mat<uint8_t> img_mat(reinterpret_cast<uint8_t*>(img.data), img.rows, img.cols);
	img_vec = arma::vectorise(arma::conv_to<arma::Mat<int>>::from(img_mat));

	return img_vec;
}

cv::Mat FrameLoader::Vector2CVMat(arma::Col<int> &img_vec, int N_row, int N_col) {

	arma::Mat<int> img_mat;

	img_mat = img_vec;

	img_mat.reshape(N_row,N_col);

	cv::Mat img(img_mat.n_rows, img_mat.n_cols, CV_32SC1, img_mat.memptr());

	img.convertTo(img, CV_8UC1);

	return img;
}

np::ndarray FrameLoader::ReturnSingleFrame(session_param &ses_par, frames &frame_now, int cam_nr) {

	int N_row = get<0>(frame_now.image_size[cam_nr]);
	int N_col = get<1>(frame_now.image_size[cam_nr]);

	p::tuple shape = p::make_tuple(N_row,N_col);
	np::dtype dtype = np::dtype::get_builtin<int>();
	np::ndarray array_out = np::zeros(shape,dtype);

	for (int i=0; i<N_row; i++) {
		for (int j=0; j<N_col; j++) {
			array_out[i][j] = frame_now.single_raw_frame[cam_nr](i*N_row+j);
		}
	}

	return array_out;
}