#ifndef VOX_GRID_H
#define VOX_GRID_H

#include <string>
#include <stdint.h>
#include <vector>
#include <map>
#include <unordered_map>
#include <armadillo>

using namespace std;

struct voxel_prop {
	int pix_0;
	vector<int> pix_n;
	int vox_ind;
	double x;
	double y;
	double z;
	vector<vector<int>> neighbors;
};

struct vox_grid {
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
	//vector<voxel_prop> voxel_list;
	multimap<int,int> pix2vox; // only for camera 0 to voxels
	unordered_map<int,vector<int>> vox2pix; //voxels to pixels for all views
};
#endif