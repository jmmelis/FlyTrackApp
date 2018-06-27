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
		
		const int max_n_seg = 20;

		FocalGrid();

		bool LoadCalibration(session_param &ses_par, vox_grid &vox, frames &frame);
		bool ConstructFocalGrid(vox_grid &vox);
		vector<tuple<double,double,double,int>> ProjectImage2Cloud(vector<arma::Col<int>> &frame_in, vox_grid &vox);
		//vector<tuple<double,double,double,int>> ProjectImage2CloudThread(int i_start, int i_end, vox_grid &vox, vector<arma::Col<int>> &frame_in);
		vector<arma::Col<int>> ProjectCloud2Image(vector<tuple<double,double,double,int>> &cloud_in, vox_grid &vox);
		//tuple<bool,int> CheckInView(vector<arma::Col<int>> &frame_in, vox_grid &vox, int vox_ind);
		tuple<bool,vector<int>> CheckInView(vector<arma::Col<int>> &frame_in, vox_grid &vox, int vox_ind);
		tuple<int,bool,vector<int>> CheckCode(vector<vector<int>> code_list, vector<int> code_in);
		inline bool CheckNeighbors(vector<arma::Col<int>> &frame_in, vox_grid &vox, int vox_ind);
		vector<voxel_prop> CheckVoxel(int i);
		inline int IsVoxel(arma::Mat<double> uv);
		arma::Mat<int> TransformXYZ2UV(arma::Col<double> xyz_pos);
		tuple<arma::Mat<int>, arma::Col<double>> RayCasting(int cam_nr, arma::Col<double> xyz_pos_prev, arma::Col<double> uv_pos_prev, arma::Col<double> uv_pos_now);
		arma::Mat<double> Camera2WorldMatrix(arma::Col<double> calib_param);
		arma::Mat<double> World2CameraMatrix(arma::Col<double> calib_param);

};
#endif