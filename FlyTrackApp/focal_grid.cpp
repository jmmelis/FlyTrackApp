#include "focal_grid.h"

#include <string>
#include <stdint.h>
#include <iostream>
#include <iomanip>
#include <math.h>
#include <chrono>
#include <vector>
#include <map>
#include <unordered_map>
#include <iterator>
#include <future>
#include <thread>
#include <algorithm>
#include <dirent.h>
#include <armadillo>

using namespace std;

FocalGrid::FocalGrid() {
	// empty
}

bool FocalGrid::LoadCalibration(session_param &ses_par, vox_grid &vox, frames &frame) {

	bool grid_loaded = true;

	try {

		vox.image_size.clear();
		vox.calib_mat.clear();
		vox.X_xyz.clear();
		vox.X_uv.clear();
		vox.uv_offset.clear();

		image_size.clear();
		calib_mat.clear();
		X_xyz.clear();
		X_uv.clear();
		uv_offset.clear();

		string calib_file = ses_par.session_loc + "/" + ses_par.cal_loc + "/" + ses_par.cal_name;

		arma::Mat<double> CalibMatrix;

		CalibMatrix.load(calib_file);

		for (int i=0; i<ses_par.N_cam; i++) {
			vox.image_size.push_back(frame.image_size[i]);
			vox.calib_mat.push_back(CalibMatrix.col(i));
			vox.X_xyz.push_back(FocalGrid::Camera2WorldMatrix(CalibMatrix.col(i)));
			vox.X_uv.push_back(FocalGrid::World2CameraMatrix(CalibMatrix.col(i)));
			arma::Col<double> uv_off_i = {CalibMatrix(10,i)/2.0-CalibMatrix(12,i)/2.0, CalibMatrix(9,i)/2.0-CalibMatrix(11,i)/2.0, 0.0};
			vox.uv_offset.push_back(uv_off_i);

			image_size.push_back(frame.image_size[i]);
			calib_mat.push_back(CalibMatrix.col(i));
			X_xyz.push_back(FocalGrid::Camera2WorldMatrix(CalibMatrix.col(i)));
			X_uv.push_back(FocalGrid::World2CameraMatrix(CalibMatrix.col(i)));
			uv_offset.push_back(uv_off_i);
		}

		// Populate internal parameters

		N_cam = vox.N_cam;
		N_threads = vox.N_threads;
		nx = vox.nx;
		ny = vox.ny;
		nz = vox.nz;
		ds = vox.ds;
		x0 = vox.x0;
		y0 = vox.y0;
		z0 = vox.z0;

	}
	catch (...) {
		grid_loaded = false;
	}

	return grid_loaded;
}

bool FocalGrid::ConstructFocalGrid(vox_grid &vox) {

	bool grid_build = true;

	int voxel_ind = 0;

	try {

		vox.pix2vox.clear();

		vox.vox2pix.clear();

		for (int k=0; k<nz; k++) {
			cout << k << endl;
			for (int j=0; j<ny; j++) {
				for (int i=0; i<nx; i++) {
					vector<int> uv_voxel = FocalGrid::CheckVoxel(i, j, k);
					if (uv_voxel.size()==N_cam) {
						voxel_ind = k*nx*ny+j*nx+i;
						vox.pix2vox.insert(pair<int,int>(uv_voxel[0],voxel_ind));
						vox.vox2pix.insert(pair<int,vector<int>>(voxel_ind,uv_voxel));
					}
				}
			}
		}

	}
	catch (...) {
		grid_build = false;
	}

	return grid_build;
}

vector<int> FocalGrid::CheckVoxel(int i, int j, int k) {

	vector<int> uv_out;

	int uv_ind = -1;

	arma::Col<double> xyz(4);

	xyz = {x0-((nx-1)/2.0)*ds+i*ds, 
		y0-((ny-1)/2.0)*ds+j*ds,
		z0-((nz-1)/2.0)*ds+k*ds,
		1.0};

	arma::Col<double> uv(3);

	int n=0;
	bool uv_out_of_range = false;

	while (n<N_cam && uv_out_of_range==false) {

		uv = X_uv[n]*xyz-uv_offset[n];

		if (uv(0)>=0 && uv(0)<(get<1>(image_size[n]))) {
			if (uv(1)>=0 && uv(1)<(get<0>(image_size[n]))) {
				uv_ind = ((int) uv(1))*get<1>(image_size[n])+((int) uv(0));
				uv_out.push_back(uv_ind);
			}
			else {
				uv_out.clear();
				uv_out_of_range = true;
			}
		}
		else {
			uv_out.clear();
			uv_out_of_range = true;
		}
		n++;
	}

	return uv_out;
}

arma::Col<double> FocalGrid::PointCloudMatching(arma::Mat<double> &dest_pcl, arma::Mat<double> &src_pcl, double search_radius) {

	// Create a multimap of the two pointclouds and try to find dest-src pairs in the x, y and z direction:

	multimap<int,double> xy_map;
	multimap<int,double> xz_map;
	multimap<int,double> yz_map;

	int N_dest = dest_pcl.n_cols;

	cout << "N_dest" << endl;
	cout << N_dest << endl;

	for (int i=0; i<N_dest; i++) {

		int i_x = (int) (dest_pcl(1,i)-x0)/ds+((nx-1)/2.0);
		int i_y = (int) (dest_pcl(2,i)-y0)/ds+((ny-1)/2.0);
		int i_z = (int) (dest_pcl(3,i)-z0)/ds+((nz-1)/2.0);

		xy_map.insert(pair<int,double>(i_y*nx+i_x,dest_pcl(1,i)));
		xz_map.insert(pair<int,double>(i_x*nz+i_z,dest_pcl(2,i)));
		yz_map.insert(pair<int,double>(i_z*ny+i_y,dest_pcl(3,i)));

	}

	int N_src = src_pcl.n_cols;

	cout << "N_src" << endl;
	cout << N_src << endl;

	arma::Col<double> src_cg;
	src_cg.zeros(3);

	vector<tuple<double,double,double>> delta_x_vec; 
	vector<tuple<double,double,double>> delta_y_vec; 
	vector<tuple<double,double,double>> delta_z_vec;

	for (int j=0; j<N_src; j++) {

		src_cg(0) += src_pcl(1,j)/(N_src*1.0);
		src_cg(1) += src_pcl(2,j)/(N_src*1.0);
		src_cg(2) += src_pcl(3,j)/(N_src*1.0);

		int j_x = (int) (src_pcl(1,j)-x0)/ds+((nx-1)/2.0);
		int j_y = (int) (src_pcl(2,j)-y0)/ds+((ny-1)/2.0);
		int j_z = (int) (src_pcl(3,j)-z0)/ds+((nz-1)/2.0);

		// delta z
		pair<multimap<int,double>::iterator, multimap<int,double>::iterator> dest_z_voxels;
		dest_z_voxels = xy_map.equal_range(j_y*nx+j_x);
		double delta_z = 100000.0;
		double delta_z_new;
		double z_val;

		for (multimap<int,double>::iterator it_z=dest_z_voxels.first; it_z != dest_z_voxels.second; ++it_z) {
			z_val = it_z->second;
			delta_z_new = z_val-src_pcl(3,j);
			if (abs(delta_z_new)<abs(delta_z)) {
				delta_z = delta_z_new;
			}
		}

		if (abs(delta_z)<search_radius) {
			delta_z_vec.push_back(make_tuple(delta_z,src_pcl(1,j),src_pcl(2,j)));
		}

		// delta y
		pair<multimap<int,double>::iterator, multimap<int,double>::iterator> dest_y_voxels;
		dest_y_voxels = xz_map.equal_range(j_x*nz+j_z);
		double delta_y = 100000.0;
		double delta_y_new;
		double y_val;

		for (multimap<int,double>::iterator it_y=dest_y_voxels.first; it_y != dest_y_voxels.second; ++it_y) {
			y_val = it_y->second;
			delta_y_new = y_val-src_pcl(2,j);
			if (abs(delta_y_new)<abs(delta_y)) {
				delta_y = delta_y_new;
			}
		}

		if (abs(delta_y)<search_radius) {
			delta_y_vec.push_back(make_tuple(delta_y,src_pcl(1,j),src_pcl(3,j)));
		}

		// delta x
		pair<multimap<int,double>::iterator, multimap<int,double>::iterator> dest_x_voxels;
		dest_x_voxels = yz_map.equal_range(j_z*ny+j_y);
		double delta_x = 100000.0;
		double delta_x_new;
		double x_val;

		for (multimap<int,double>::iterator it_x=dest_x_voxels.first; it_x != dest_x_voxels.second; ++it_x) {
			x_val = it_x->second;
			delta_x_new = x_val-src_pcl(1,j);
			if (abs(delta_x_new)<abs(delta_x)) {
				delta_x = delta_x_new;
			}
		}

		if (abs(delta_x)<search_radius) {
			delta_x_vec.push_back(make_tuple(delta_x,src_pcl(2,j),src_pcl(3,j)));
		}

	}

	double theta_x = 0.0;
	double trans_x = 0.0;
	double theta_y = 0.0;
	double trans_y = 0.0;
	double theta_z = 0.0;
	double trans_z = 0.0;

	int N_dx = delta_x_vec.size();
	int N_dy = delta_y_vec.size();
	int N_dz = delta_z_vec.size();

	cout << "delta x size" << endl;
	cout << N_dx << endl;
	cout << "delta y size" << endl;
	cout << N_dy << endl;
	cout << "delta z size" << endl;
	cout << N_dz << endl;

	double mse_x = 0.0;
	double mse_y = 0.0;
	double mse_z = 0.0;

	double theta_x_update;
	double theta_y_update;
	double theta_z_update;
	double trans_x_update;
	double trans_y_update;
	double trans_z_update;

	int N_theta_x = 0;
	int N_theta_y = 0;
	int N_theta_z = 0;
	int N_trans_x = 0;
	int N_trans_y = 0;
	int N_trans_z = 0;

	for (int k=0; k<N_dx; k++) {

		trans_x_update = get<0>(delta_x_vec[k]);
		theta_y_update = atan2(get<0>(delta_x_vec[k]),get<2>(delta_x_vec[k])-src_cg(2));
		theta_z_update = atan2(get<0>(delta_x_vec[k]),get<1>(delta_x_vec[k])-src_cg(1));

		if (isfinite(trans_x_update)==true) {
			trans_x += trans_x_update;
			mse_x += pow(get<0>(delta_x_vec[k]),2);
			N_trans_x++;
		}
		if (isfinite(theta_y_update)==true) {
			theta_y += theta_y_update;
			N_theta_y++;
		}
		if (isfinite(theta_z_update)==true) {
			theta_z -= theta_z_update;
			N_theta_z++;
		}
	}

	for (int k=0; k<N_dy; k++) {

		trans_y_update = get<0>(delta_y_vec[k]);
		theta_x_update = atan2(get<0>(delta_y_vec[k]),get<2>(delta_y_vec[k])-src_cg(2));
		theta_z_update = atan2(get<0>(delta_y_vec[k]),get<2>(delta_y_vec[k])-src_cg(0));

		if (isfinite(trans_y_update)==true) {
			trans_y += trans_y_update;
			mse_y += pow(get<0>(delta_y_vec[k]),2);
			N_trans_y++;
		}
		if (isfinite(theta_x_update)==true) {
			theta_x -= theta_x_update;
			N_theta_x++;
		}
		if (isfinite(theta_z_update)==true) {
			theta_z += theta_z_update;
			N_theta_z++;
		}
	}

	for (int k=0; k<N_dz; k++) {

		trans_z_update = get<0>(delta_z_vec[k]);
		theta_x_update = atan2(get<0>(delta_z_vec[k]),get<2>(delta_z_vec[k])-src_cg(1));
		theta_y_update = atan2(get<0>(delta_z_vec[k]),get<1>(delta_z_vec[k])-src_cg(0));

		if (isfinite(trans_z_update)==true) {
			trans_z += trans_z_update;
			mse_z += pow(get<0>(delta_z_vec[k]),2);
			N_trans_z++;
		}
		if (isfinite(theta_x_update)==true) {
			theta_x += theta_x_update;
			N_theta_x++;
		}
		if (isfinite(theta_y_update)==true) {
			theta_y -= theta_y_update;
			N_theta_y++;
		}
	}

	arma::Col<double> state_update(9);

	if (N_theta_x>3) {
		state_update(0) = theta_x/(N_theta_x*1.0);
	}
	else {
		state_update(0) = 0.0;
	}
	if (N_theta_y>3) {
		state_update(1) = theta_y/(N_theta_y*1.0);
	}
	else {
		state_update(1) = 0.0;
	}
	if (N_theta_z>3) {
		state_update(2) = theta_z/(N_theta_z*1.0);
	}
	else {
		state_update(2) = 0.0;
	}
	if (N_trans_x>3) {
		state_update(3) = trans_x/(N_trans_x*1.0);
		state_update(6) = mse_x/(N_trans_x*1.0);
	}
	else {
		state_update(3) = 0.0;
		state_update(6) = 1.0;
	}
	if (N_trans_y>3) {
		state_update(4) = trans_y/(N_trans_y*1.0);
		state_update(7) = mse_y/(N_trans_y*1.0);
	}
	else {
		state_update(4) = 0.0;
		state_update(7) = 1.0;
	}
	if (N_trans_z>3) {
		state_update(5) = trans_z/(N_trans_z*1.0);
		state_update(8) = mse_z/(N_trans_z*1.0);
	}
	else {
		state_update(5) = 0.0;
		state_update(8) = 1.0;
	}

	cout << "state update" << endl;
	cout << state_update << endl;

	return state_update;
}

vector<tuple<int,double,double,double,double,double,double>> FocalGrid::ProjectImage2Cloud(vector<arma::Col<int>> &frame_in, vox_grid &vox) {

	//vector<tuple<double,double,double,int>> pcl_now;

	vector<tuple<int,double,double,double,double,double,double>> pcl_now;

	unordered_map<int,int> pcl_voxels;

	int N_row = get<0>(vox.image_size[0]);
	int N_col = get<1>(vox.image_size[0]);

	pair<multimap<int,int>::iterator, multimap<int,int>::iterator> voxels_i;

	int vox_now;
	vector<int> uv_now;

	int n = 0;
	bool is_voxel = true;

	int frame_val_0 = 0;
	int frame_val_n = 0;

	int code_now = 0;

	int count = 0;

	// Insert voxels into an unordered map and give them a segment code:
	for (int i=0; i<(N_row*N_col); i++) {
		frame_val_0 = frame_in[0](i);
		if (frame_val_0>0) {
			voxels_i = vox.pix2vox.equal_range(i);
			for (multimap<int,int>::iterator it=voxels_i.first; it != voxels_i.second; ++it) {
				vox_now = it->second;
				uv_now = vox.vox2pix[vox_now];
				n = 1;
				is_voxel = true;
				code_now = frame_val_0;
				while (is_voxel==true && n < N_cam) {
					frame_val_n = frame_in[n](uv_now[n]);
					if (frame_val_n>0) {
						code_now = code_now+pow(max_n_seg,n)*frame_val_n;
					}
					else {
						is_voxel = false;
					}
					n++;
				}
				if (is_voxel==true) {
					count++;
					pcl_voxels.insert(pair<int,int>(vox_now,code_now));
				}
			}
		}
	}

	// For each voxel in the pcl_voxels map:
	// -> find the neighboring voxels
	// -> calculate normal
	// -> if a normal could be calculated, add the voxel_id, xyz_pos and normal to pcl_now

	unordered_map<int,int>::iterator nb_it;

	arma::Col<int> neighbors(27);

	arma::Col<double> nb_vec(27);

	arma::Col<double> normal(3);

	arma::Col<double> xyz_pos(3);

	int nb_vox_id;
	int nb_vox_code;

	for (nb_it = pcl_voxels.begin(); nb_it != pcl_voxels.end(); nb_it++) {
		vox_now = nb_it->first;
		code_now = nb_it->second;
		neighbors = FocalGrid::FindNeighbors(vox_now);
		if (neighbors(13)>0) {
			for (int m=0; m<27; m++) {
				if (pcl_voxels.find(neighbors(m)) != pcl_voxels.end()) {
					nb_vec(m) = 0.0;
				}
				else {
					nb_vec(m) = 1.0;
				}
			}
			normal = FocalGrid::CalculateNormal(nb_vec);
			if (arma::norm(normal,2)>0.5) {
				xyz_pos = FocalGrid::CalculatePosition(vox_now);
				pcl_now.push_back(make_tuple(code_now,xyz_pos(0),xyz_pos(1),xyz_pos(2),normal(0),normal(1),normal(2)));
			}
		}
	}

	if (pcl_now.size() <= 0) {
		pcl_now.push_back(make_tuple(0,0.0,0.0,0.0,0.0,0.0,0.0));
	}
	return pcl_now;
}

arma::Col<int> FocalGrid::FindNeighbors(int vox_ind) {

	// find the neighboring voxels:

	arma::Col<int> vox_int_mat(27);

	int i = vox_ind % nx;
	int j = ((vox_ind-i)/nx) % ny;
	int k = (vox_ind-i-j*nx)/(nx*ny);

	if ((i>0 && i<(nx-1)) && (j>0 && j<(ny-1)) && (k>0 && k<(nz-1))) {

		vox_int_mat(0) = vox_ind-nx*ny-nx-1;
		vox_int_mat(1) = vox_ind-nx*ny-nx;
		vox_int_mat(2) = vox_ind-nx*ny-nx+1;
		vox_int_mat(3) = vox_ind-nx*ny-1;
		vox_int_mat(4) = vox_ind-nx*ny;
		vox_int_mat(5) = vox_ind-nx*ny+1;
		vox_int_mat(6) = vox_ind-nx*ny+nx-1;
		vox_int_mat(7) = vox_ind-nx*ny+nx;
		vox_int_mat(8) = vox_ind-nx*ny+nx+1;
		vox_int_mat(9) = vox_ind-nx-1;
		vox_int_mat(10) = vox_ind-nx;
		vox_int_mat(11) = vox_ind-nx+1;
		vox_int_mat(12) = vox_ind-1;
		vox_int_mat(13) = vox_ind;
		vox_int_mat(14) = vox_ind+1;
		vox_int_mat(15) = vox_ind+nx-1;
		vox_int_mat(16) = vox_ind+nx;
		vox_int_mat(17) = vox_ind+nx+1;
		vox_int_mat(18) = vox_ind+nx*ny-nx-1;
		vox_int_mat(19) = vox_ind+nx*ny-nx;
		vox_int_mat(20) = vox_ind+nx*ny-nx+1;
		vox_int_mat(21) = vox_ind+nx*ny-1;
		vox_int_mat(22) = vox_ind+nx*ny;
		vox_int_mat(23) = vox_ind+nx*ny+1;
		vox_int_mat(24) = vox_ind+nx*ny+nx-1;
		vox_int_mat(25) = vox_ind+nx*ny+nx;
		vox_int_mat(26) = vox_ind+nx*ny+nx+1;

	}
	else {
		vox_int_mat.zeros();
	}

	return vox_int_mat;
}

arma::Col<double> FocalGrid::CalculatePosition(int vox_ind) {

	int i = vox_ind % nx;
	int j = ((vox_ind-i)/nx) % ny;
	int k = (vox_ind-i-j*nx)/(nx*ny);

	arma::Col<double> xyz_pos(3);

	xyz_pos(0) = x0-((nx-1)/2.0)*ds+i*ds;
	xyz_pos(1) = y0-((ny-1)/2.0)*ds+j*ds;
	xyz_pos(2) = z0-((nz-1)/2.0)*ds+k*ds;

	return xyz_pos;
}

arma::Col<double> FocalGrid::CalculateNormal(arma::Col<double> neighbor_vector) {

	arma::Col<double> normal;

	normal.zeros(3);

	if (arma::sum(neighbor_vector)>1.0 && arma::sum(neighbor_vector)<20.0) {
		normal = normal_mat.t()*neighbor_vector;
		if (arma::norm(normal,2)>0.5){
			normal = arma::normalise(normal);
		}
		else {
			normal.zeros(3);
		}
	}

	return normal;
}

vector<arma::Col<int>> FocalGrid::ProjectCloud2Image(vector<tuple<double,double,double,int>> &cloud_in) {

	vector<arma::Col<int>> frame_now;

	int N_vox = cloud_in.size();

	for (int n=0; n<N_cam; n++) {
		arma::Col<int> frame_n;
		frame_n.zeros(get<0>(image_size[n])*get<1>(image_size[n]));
		frame_now.push_back(frame_n);
	}

	arma::Col<double> uv;
	arma::Col<double> xyz;
	int u = 0;
	int v = 0;

	for (int i=0; i<N_vox; i++) {
		for (int j=0; j<N_cam; j++) {
			xyz = {get<0>(cloud_in[i]),get<1>(cloud_in[i]),get<2>(cloud_in[i]),1.0};
			uv = X_uv[j]*xyz-uv_offset[j];
			if (uv(0)>=0 && uv(0)<(get<1>(image_size[j]))) {
				if (uv(1)>=0 && uv(1)<(get<0>(image_size[j]))) {
					u = (int) uv(0);
					v = (int) uv(1);
					if (frame_now[j](get<1>(image_size[j])*v+u)==0) {
						frame_now[j](get<1>(image_size[j])*v+u) = get<3>(cloud_in[i]);
					}
					else {
						if (frame_now[j](get<1>(image_size[j])*v+u)>get<3>(cloud_in[i])) {
							frame_now[j](get<1>(image_size[j])*v+u) = get<3>(cloud_in[i]);
						}
					}
				}
			}
		}
	}
	return frame_now;
}

arma::Mat<int> FocalGrid::TransformXYZ2UV(arma::Col<double> xyz_pos) {

	arma::Mat<int> uv_mat(2,N_cam);

	for (int n=0; n<N_cam; n++) {
		arma::Col<double> uv = X_uv[n]*xyz_pos-uv_offset[n];
		uv_mat(0,n) = int(uv(0));
		uv_mat(1,n) = int(uv(1));
	}

	return uv_mat;
}

tuple<arma::Mat<int>, arma::Col<double>> FocalGrid::RayCasting(int cam_nr, arma::Col<double> xyz_pos_prev, arma::Col<double> uv_pos_prev, arma::Col<double> uv_pos_now) {

	// Calculate the 3D translation vector:
	arma::Col<double> xyz_uv_prev = X_xyz[cam_nr]*(uv_pos_prev+uv_offset[cam_nr]);
	arma::Col<double> xyz_uv_now = X_xyz[cam_nr]*(uv_pos_now+uv_offset[cam_nr]);
	arma::Col<double> trans_vec = xyz_uv_now-xyz_uv_prev;
	trans_vec(3) = 0.0;

	// Add the translation to the xyz position:
	arma::Col<double> xyz_pos_now = xyz_pos_prev + trans_vec;

	// Project the new position back to the camera views:

	arma::Mat<int> uv_mat(3,N_cam);

	for (int n=0; n<N_cam; n++) {
		arma::Col<double> uv = X_uv[n]*xyz_pos_now-uv_offset[n];
		if (uv(0)>=0 && uv(0)<(get<1>(image_size[n])) && uv(1)>=0 && uv(1)<(get<0>(image_size[n]))) {
			uv_mat(0,n) = int(uv(0));
			uv_mat(1,n) = int(uv(1));
			uv_mat(2,n) = int(uv(2));
		}
		else {
			arma::Col<double> uv_old = X_uv[n]*xyz_pos_prev-uv_offset[n];
			uv_mat(0,n) = int(uv_old(0));
			uv_mat(1,n) = int(uv_old(1));
			uv_mat(2,n) = int(uv_old(2));
		}
	}

	return make_tuple(uv_mat,xyz_pos_now);
}

arma::Mat<double> FocalGrid::Camera2WorldMatrix(arma::Col<double> calib_param) {

	// return the world to camera projection matrix

	arma::Mat<double> C = {{calib_param(0), calib_param(2), 0, 0},
		 			{0, calib_param(1), 0, 0},
		 			{0, 0, 0, 1}};

	double theta = sqrt(pow(calib_param(3),2)+pow(calib_param(4),2)+pow(calib_param(5),2));

	arma::Mat<double> omega = {{0, -calib_param(5), calib_param(4)},
			 			{calib_param(5), 0, -calib_param(3)},
			 			{-calib_param(4), calib_param(3), 0}};

	arma::Mat<double> R(3,3); R.eye();

	R = R+(sin(theta)/theta)*omega+((1-cos(theta))/pow(theta,2))*(omega*omega);

	arma::Col<double> T = {calib_param(6), calib_param(7), calib_param(8)};

	arma::Mat<double> K = {{R(0,0), R(0,1), R(0,2), T(0)},
		 				{R(1,0), R(1,1), R(1,2), T(1)},
		 				{R(2,0), R(2,1), R(2,2), T(2)},
		 				{0, 0, 0, 1}};

	return arma::inv(K)*arma::pinv(C);

}

arma::Mat<double> FocalGrid::World2CameraMatrix(arma::Col<double> calib_param) {

	// return the world to camera projection matrix

	arma::Mat<double> C = {{calib_param(0), calib_param(2), 0, 0},
		 			{0, calib_param(1), 0, 0},
		 			{0, 0, 0, 1}};

	double theta = sqrt(pow(calib_param(3),2)+pow(calib_param(4),2)+pow(calib_param(5),2));

	arma::Mat<double> omega = {{0, -calib_param(5), calib_param(4)},
			 			{calib_param(5), 0, -calib_param(3)},
			 			{-calib_param(4), calib_param(3), 0}};

	arma::Mat<double> R(3,3); R.eye();

	R = R+(sin(theta)/theta)*omega+((1-cos(theta))/pow(theta,2))*(omega*omega);

	arma::Col<double> T = {calib_param(6), calib_param(7), calib_param(8)};

	arma::Mat<double> K = {{R(0,0), R(0,1), R(0,2), T(0)},
		 				{R(1,0), R(1,1), R(1,2), T(1)},
		 				{R(2,0), R(2,1), R(2,2), T(2)},
		 				{0, 0, 0, 1}};

	return C*K;

}