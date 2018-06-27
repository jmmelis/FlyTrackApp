#ifndef FRAMES_H
#define FRAMES_H

#include <string>
#include <stdint.h>
#include <vector>
#include <armadillo>

using namespace std;

struct frames {
	int batch_nr;
	int start_frame;
	int end_frame;
	vector<tuple<int, int>> image_size;
	vector<arma::Col<int>> bckg_images;
	vector<arma::Col<int>> single_raw_frame;
	vector<arma::Mat<int>> raw_frames;
	vector<arma::Col<int>> single_seg_frame;
	vector<arma::Mat<int>> seg_frames;
	vector<tuple<int,double,double,double,double,double,double>> single_frame_pcl;
	vector<vector<tuple<int,double,double,double,double,double,double>>> frame_batch_pcl;
	vector<arma::Mat<double>> pcl_init_single_frame;
	vector<arma::Mat<double>> M_init_single_frame;
	vector<vector<arma::Mat<double>>> pcl_init_frame_batch;
	vector<vector<arma::Mat<double>>> M_init_frame_batch;
};
#endif