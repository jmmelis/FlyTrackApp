#ifndef FOCAL_GRID_CLASS_H
#define FOCAL_GRID_CLASS_H

#include "session_param.h"
#include "frames.h"
#include "vox_grid.h"

#include <string>
#include <stdint.h>
#include <iostream>
#include <iomanip>
#include <math.h>
#include <chrono>
#include <vector>
#include <map>
#include <unordered_map>
#include <future>
#include <thread>
#include <algorithm>
#include <dirent.h>
#include <armadillo>

using namespace std;

class FocalGrid {

	public:

		int N_cam;
		int N_threads;
		int nx;
		int ny;
		int nz;
		double ds;
		double x0;
		double y0;
		double z0;
		vector<tuple<int,int>> image_size;
		vector<arma::Col<double>> calib_mat;
		vector<arma::Mat<double>> X_xyz;
		vector<arma::Mat<double>> X_uv;
		vector<arma::Col<double>> uv_offset;
		
		const int max_n_seg = 100;

		double a0 = 0.0;
		double a1 = 1.0;
		double a2 = sqrt(2.0)/2.0;
		double a3 = sqrt(3.0)/3.0;

		const arma::Mat<double> normal_mat = {
			{-a3,-a3,-a3},
			{a0,-a2,-a2},
			{a3,-a3,-a3},
			{-a2,a0,-a2},
			{a0,a0,-a1},
			{a2,a0,-a2},
			{-a3,a3,-a3},
			{a0,a2,-a2},
			{a3,a3,-a3},
			{-a2,-a2,a0},
			{a0,-a1,a0},
			{a2,-a2,a0},
			{-a1,a0,a0},
			{a0,a0,a0},
			{a1,a0,a0},
			{-a2,a2,a0},
			{a0,a1,a0},
			{a2,a2,a0},
			{-a3,-a3,a3},
			{a0,-a2,a2},
			{a3,-a3,a3},
			{-a2,a0,a2},
			{a0,a0,a1},
			{a2,a0,a2},
			{-a3,a3,a3},
			{a0,a2,a2},
			{a3,a3,a3}};

		FocalGrid();

		bool LoadCalibration(session_param &ses_par, vox_grid &vox, frames &frame);
		bool ConstructFocalGrid(vox_grid &vox);
		vector<int> CheckVoxel(int i, int j, int k);
		arma::Col<double> PointCloudMatching(arma::Mat<double> &dest_pcl, arma::Mat<double> &src_pcl, double search_radius);
		vector<tuple<int,double,double,double,double,double,double>> ProjectImage2Cloud(vector<arma::Col<int>> &frame_in, vox_grid &vox);
		vector<arma::Col<int>> ProjectCloud2Image(vector<tuple<double,double,double,int>> &cloud_in);
		arma::Col<int> FindNeighbors(int vox_ind);
		arma::Col<double> CalculatePosition(int vox_ind);
		arma::Col<double> CalculateNormal(arma::Col<double> neighbor_vector);
		arma::Mat<int> TransformXYZ2UV(arma::Col<double> xyz_pos);
		tuple<arma::Mat<int>, arma::Col<double>> RayCasting(int cam_nr, arma::Col<double> xyz_pos_prev, arma::Col<double> uv_pos_prev, arma::Col<double> uv_pos_now);
		arma::Mat<double> Camera2WorldMatrix(arma::Col<double> calib_param);
		arma::Mat<double> World2CameraMatrix(arma::Col<double> calib_param);

};
#endif