#include "focal_grid.h"

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

//-------------------------------------------------

inline bool sortByPix0(const voxel_prop &lhs, const voxel_prop &rhs) {
	return lhs.pix_0 < rhs.pix_0;
}

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

	try {

		vox.voxel_list.clear();

		int vec_size = 0;
		int i = 0;
		vector<future<vector<voxel_prop>>> vox_results;
		
		while (i<nx) {
			cout << "row " + to_string(i) + " / " + to_string(nx) << endl;
			vox_results.clear();
			for (int j=0; j<N_threads; j++) {
				if (i+j<nx) {
					vox_results.push_back(async(launch::async, &FocalGrid::CheckVoxel,this,i+j));
				}
			}
			for (int j=0; j<N_threads; j++) {
				vector<voxel_prop> vox_results_thread=vox_results.at(j).get();
				vec_size = vox_results_thread.size();
				if (vec_size>1) {
					copy(vox_results_thread.begin(),vox_results_thread.end(),back_inserter(vox.voxel_list));
				}
			}
			i+=N_threads;
		}

		cout << "Sorting" << endl;
		sort(vox.voxel_list.begin(),vox.voxel_list.end(),sortByPix0);

		cout << "Voxel list size:" << endl;
		cout << vox.voxel_list.size() << endl;

	}
	catch (...) {
		grid_build = false;
	}

	return grid_build;
}

vector<tuple<double,double,double,int>> FocalGrid::ProjectImage2Cloud(vector<arma::Col<int>> &frame_in, vox_grid &vox) {

	vector<tuple<double,double,double,int>> pcl_now;

	tuple<bool,vector<int>> view_check;
	tuple<int,bool,vector<int>> code_check;
	vector<vector<int>> code_list;

	int N_row = get<0>(vox.image_size[0]);
	int N_col = get<1>(vox.image_size[0]);

	int vox_ind = 0;
	int max_ind = vox.voxel_list.size();

	for (int i=0; i<(N_row*N_col); i++) {

		if (frame_in[0](i)>0) {
			while (vox.voxel_list[vox_ind].pix_0<=i) {
				if (vox.voxel_list[vox_ind].pix_0==i) {
					view_check = FocalGrid::CheckInView(frame_in, vox, vox_ind);
					if (get<0>(view_check)==true) {
						if (FocalGrid::CheckNeighbors(frame_in, vox, vox_ind)==true) {
							code_check = FocalGrid::CheckCode(code_list, get<1>(view_check));
							if (get<1>(code_check)==true) {
								code_list.push_back(get<2>(code_check));
							}
							pcl_now.push_back(make_tuple(
								vox.voxel_list[vox_ind].x,
								vox.voxel_list[vox_ind].y,
								vox.voxel_list[vox_ind].z,
								get<0>(code_check)));
								//get<1>(view_check)));
						}
					}
				}
				vox_ind++;
				if (vox_ind >= max_ind) {
					break;
				}
			}
			if (vox_ind >= max_ind) {
				break;
			}
		}
	}

	if (pcl_now.size() <= 0) {
		pcl_now.push_back(make_tuple(0.0,0.0,0.0,0));
	}

	return pcl_now;
}

tuple<int,bool,vector<int>> FocalGrid::CheckCode(vector<vector<int>> code_list, vector<int> code_in) {

	int code_out = 0;
	vector<int> new_code;

	int i=0;
	bool code_found = false;
	bool add_code = false;
	
	int N_codes = code_list.size();

	if (N_codes > 0) {
		while ((i<N_codes) && (code_found==false)) {
			int match_count = 0;
			int body_count = 0;
			for (int j=0; j<N_cam; j++) {
				if (code_list[i][j]==code_in[j]) {
					if (code_in[j]==1) {
						match_count++;
						body_count++;
					}
					else {
						match_count++;
					}
				}
			}
			if (match_count == N_cam) {
				// Code exists, get code number
				code_out = code_list[i][N_cam+1];
				code_found == true;
			}
			else if ((match_count == (N_cam-1)) && (body_count < (N_cam-1))) {
				// Code cannot be found in code_list but it matches with N_cam-1 views
				// and has less than N_cam-1 body views:
				code_out = code_list[i][N_cam+1];
				code_found == true;
			}
			i++;
		}
		if (code_found == false) {
			// Code could not be found in the code list, add code to the list.
			add_code = true;

			// Check first if code_in corresponds to the body:
			int body_count = 0;
			for (int j=0; j<N_cam; j++) {
				if (code_in[j]==1) {
					body_count++;
				}
				new_code.push_back(code_in[j]);
			}

			// If code_in corresponds to the body: pick 1 as pcl code
			if (body_count == N_cam) {
				new_code.push_back(1);
				code_out = 1;
			}
			// If code_in does not correspond to the body: add N_codes+1 as pcl code
			else {
				new_code.push_back(N_codes+1);
				code_out = N_codes+1;
			}
		}
	}
	else {
		// No code exists yet, add curent code to the code list:
		add_code = true;

		// Check first if code_in corresponds to the body:
		int body_count = 0;
		for (int j=0; j<N_cam; j++) {
			if (code_in[j]==1) {
				body_count++;
			}
			new_code.push_back(code_in[j]);
		}

		// If code_in corresponds to the body: pick 1 as pcl code
		if (body_count == N_cam) {
			new_code.push_back(1);
			code_out = 1;
		}
		// If code_in does not correspond to the body: add 2 as pcl code
		else {
			new_code.push_back(2);
			code_out = 2;
		}
	}

	return make_tuple(code_out,add_code,new_code);
}

/*
vector<tuple<double,double,double,int>> FocalGrid::ProjectImage2Cloud(vector<arma::Col<int>> &frame_in, vox_grid &vox) {

	vector<tuple<double,double,double,int>> pcl_now;

	vector<future<vector<tuple<double,double,double,int>>>> future_pcl;

	int N_row = get<0>(vox.image_size[0]);
	int N_col = get<1>(vox.image_size[0]);

	int vec_size = 0;

	// Launch threads:
	for (int j=0; j<N_threads; j++) {
		future_pcl.push_back(async(launch::async, &FocalGrid::ProjectImage2CloudThread,this,j,(N_row*N_col),ref(vox),ref(frame_in)));
	}
	cout << "threads have been launched" << endl;
	// Retrieve information from threads:
	for (int j=0; j<N_threads; j++) {
		vector<tuple<double,double,double,int>> pcl_results_thread=future_pcl.at(j).get();
		vec_size = pcl_results_thread.size();
		if (pcl_results_thread.size()>1) {
			copy(pcl_results_thread.begin(),pcl_results_thread.end(),back_inserter(pcl_now));
		}
	}
	cout << "pcl now size" << endl;
	cout << pcl_now.size() << endl;
	if (pcl_now.size() <= 0) {
		pcl_now.push_back(make_tuple(0.0,0.0,0.0,0));
	}

	return pcl_now;
}


vector<tuple<double,double,double,int>> FocalGrid::ProjectImage2CloudThread(int i_start, int i_end, vox_grid &vox, vector<arma::Col<int>> &frame_in) {

	vector<tuple<double,double,double,int>> pcl_i;

	for (int i = i_start; i<i_end; i+=N_threads) {

		int vox_ind = 0;
		int max_ind = vox.voxel_list.size();

		tuple<bool,int> view_check;

		if (frame_in[0](i) > 0) {
			while (vox.voxel_list[vox_ind].pix_0<=i) {
				if (vox.voxel_list[vox_ind].pix_0==i) {
					view_check = FocalGrid::CheckInView(frame_in, vox, vox_ind);
					if (get<0>(view_check)==true) {
						if (FocalGrid::CheckNeighbors(frame_in, vox, vox_ind)==true) {
							pcl_i.push_back(make_tuple(
								vox.voxel_list[vox_ind].x,
								vox.voxel_list[vox_ind].y,
								vox.voxel_list[vox_ind].z,
								get<1>(view_check)));
						}
					}
				}
				vox_ind++;
				if (vox_ind >= max_ind) {
					break;
				}
			}
		}
	}

	return pcl_i;
}
*/

vector<arma::Col<int>> FocalGrid::ProjectCloud2Image(vector<tuple<double,double,double,int>> &cloud_in, vox_grid &vox) {

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
					if (frame_now[j](get<0>(image_size[j])*u+v)==0) {
						frame_now[j](get<0>(image_size[j])*u+v) = get<3>(cloud_in[i]);
					}
					else {
						if (frame_now[j](get<0>(image_size[j])*u+v)>get<3>(cloud_in[i])) {
							frame_now[j](get<0>(image_size[j])*u+v) = get<3>(cloud_in[i]);
						}
					}
				}
			}
		}
	}

	return frame_now;
}

tuple<bool,vector<int>> FocalGrid::CheckInView(vector<arma::Col<int>> &frame_in, vox_grid &vox, int vox_ind) {

	bool in_view = true;
	vector<int> seg_codes;

	int pix_ind = 0;
	int intensity_now = 0;

	for (int i=0; i<N_cam; i++) {
		if (i==0) {
			pix_ind = vox.voxel_list[vox_ind].pix_0;
			intensity_now = frame_in[i](pix_ind);
			seg_codes.push_back(intensity_now);
			if (intensity_now<=0) {
				in_view = false;
			}
		}
		else {
			pix_ind = vox.voxel_list[vox_ind].pix_n[i-1];
			intensity_now = frame_in[i](pix_ind);
			seg_codes.push_back(intensity_now);
			if (intensity_now<=0) {
				in_view = false;
			}
		}
	}

	return make_tuple(in_view,seg_codes);
}

/*
tuple<bool,int> FocalGrid::CheckInView(vector<arma::Col<int>> &frame_in, vox_grid &vox, int vox_ind) {

	int intensity_sum = 0;
	int intensity_now = 0;
	bool in_view = true;
	int n = 0;
	int pix_ind = 0;

	while (in_view==true && n<N_cam) {
		if (n==0) {
			pix_ind = vox.voxel_list[vox_ind].pix_0;
			intensity_now = frame_in[n](pix_ind);
			if (intensity_now>0) {
				if (intensity_now<max_n_seg) {
					intensity_sum += frame_in[n](pix_ind);
				}
				else {
					intensity_sum = pow((max_n_seg*1.0),N_cam);
				}
			}
			else {
				in_view = false;
			}
		}
		else {
			pix_ind = vox.voxel_list[vox_ind].pix_n[n-1];
			intensity_now = frame_in[n](pix_ind);
			if (intensity_now>0) {
				if (intensity_now<max_n_seg) {
					intensity_sum += frame_in[n](pix_ind)*pow((max_n_seg*1.0),n);
				}
				else {
					intensity_sum = pow((max_n_seg*1.0),N_cam);
				}
			}
			else {
				in_view = false;
			}
		}
		n++;
	}

	return make_tuple(in_view,intensity_sum);
}
*/

inline bool FocalGrid::CheckNeighbors(vector<arma::Col<int>> &frame_in, vox_grid &vox, int vox_ind) {

	bool pixel_on = true;

	int uv;
	int on_count = 0;
	int in_view = 0;

	for (int m=0; m<6; m++) {
		in_view =0;
		for (int n=0; n<N_cam; n++) {
			uv = vox.voxel_list[vox_ind].neighbors[m][n];
			if (frame_in[n](uv)>0) {
				in_view++;
			}
		}
		if (in_view == N_cam) {
			on_count++;
		}
	}

	if (on_count==6) {
		pixel_on = false;
	}

	if (on_count<2) {
		pixel_on = false;
	}

	return pixel_on;
}

vector<voxel_prop> FocalGrid::CheckVoxel(int i) {

	vector<voxel_prop> voxel_array;

	voxel_array.resize(1);

	int vox_index = 0;

	int is_neighbor = 0;

	arma::Col<double> xyz(4);
	arma::Col<double> xyz_0(4);
	arma::Col<double> xyz_1(4);
	arma::Col<double> xyz_2(4);
	arma::Col<double> xyz_3(4);
	arma::Col<double> xyz_4(4);
	arma::Col<double> xyz_5(4);

	arma::Mat<double> uv(3,N_cam);
	arma::Mat<double> uv_0(3,N_cam);
	arma::Mat<double> uv_1(3,N_cam);
	arma::Mat<double> uv_2(3,N_cam);
	arma::Mat<double> uv_3(3,N_cam);
	arma::Mat<double> uv_4(3,N_cam);
	arma::Mat<double> uv_5(3,N_cam);

	for (int j=0; j<ny; j++) {
		for (int k=0; k<nz; k++) {

			vox_index = i*(ny*nz)+j*nz+k;

			struct voxel_prop vox_now;

			xyz = {x0-((nx-1)/2.0)*ds+i*ds, 
				y0-((ny-1)/2.0)*ds+j*ds,
				z0-((nz-1)/2.0)*ds+k*ds,
				1.0};

			// Calculate projection coordinates:
			for (int n=0; n<N_cam; n++) {
				uv.col(n) = X_uv[n]*xyz-uv_offset[n];
			}

			// Check if the voxel projects to all frames:
			if (FocalGrid::IsVoxel(uv)==1) {
				
				xyz_0 = {x0-((nx-1)/2.0)*ds+(i+1)*ds, 
					y0-((ny-1)/2.0)*ds+j*ds,
					z0-((nz-1)/2.0)*ds+k*ds,
					1.0};

				xyz_1 = {x0-((nx-1)/2.0)*ds+(i-1)*ds, 
					y0-((ny-1)/2.0)*ds+j*ds,
					z0-((nz-1)/2.0)*ds+k*ds,
					1.0};

				xyz_2 = {x0-((nx-1)/2.0)*ds+i*ds, 
					y0-((ny-1)/2.0)*ds+(j+1)*ds,
					z0-((nz-1)/2.0)*ds+k*ds,
					1.0};

				xyz_3 = {x0-((nx-1)/2.0)*ds+i*ds, 
					y0-((ny-1)/2.0)*ds+(j-1)*ds,
					z0-((nz-1)/2.0)*ds+k*ds,
					1.0};

				xyz_4 = {x0-((nx-1)/2.0)*ds+i*ds, 
					y0-((ny-1)/2.0)*ds+j*ds,
					z0-((nz-1)/2.0)*ds+(k+1)*ds,
					1.0};

				xyz_5 = {x0-((nx-1)/2.0)*ds+i*ds, 
					y0-((ny-1)/2.0)*ds+j*ds,
					z0-((nz-1)/2.0)*ds+(k-1)*ds,
					1.0};

				for (int m=0; m<N_cam; m++) {
					uv_0.col(m) = X_uv[m]*xyz_0-uv_offset[m];
					uv_1.col(m) = X_uv[m]*xyz_1-uv_offset[m];
					uv_2.col(m) = X_uv[m]*xyz_2-uv_offset[m];
					uv_3.col(m) = X_uv[m]*xyz_3-uv_offset[m];
					uv_4.col(m) = X_uv[m]*xyz_4-uv_offset[m];
					uv_5.col(m) = X_uv[m]*xyz_5-uv_offset[m];
				}
				
				is_neighbor = FocalGrid::IsVoxel(uv_0)+FocalGrid::IsVoxel(uv_1)+
							FocalGrid::IsVoxel(uv_2)+FocalGrid::IsVoxel(uv_3)+
							FocalGrid::IsVoxel(uv_4)+FocalGrid::IsVoxel(uv_5);

				if (is_neighbor == 6) {
					for (int m=0; m<N_cam; m++) {
						if (m==0) {
							vox_now.pix_0 = ((int) uv(1,m))*get<1>(image_size[m])+((int) uv(0,m));
							vox_now.vox_ind = vox_index;
							vox_now.x = xyz(0);
							vox_now.y = xyz(1);
							vox_now.z = xyz(2);
							vox_now.neighbors.resize(6);
							vox_now.neighbors[0].push_back(((int) uv_0(1,m))*get<1>(image_size[m])+((int) uv_0(0,m)));
							vox_now.neighbors[1].push_back(((int) uv_1(1,m))*get<1>(image_size[m])+((int) uv_1(0,m)));
							vox_now.neighbors[2].push_back(((int) uv_2(1,m))*get<1>(image_size[m])+((int) uv_2(0,m)));
							vox_now.neighbors[3].push_back(((int) uv_3(1,m))*get<1>(image_size[m])+((int) uv_3(0,m)));
							vox_now.neighbors[4].push_back(((int) uv_4(1,m))*get<1>(image_size[m])+((int) uv_4(0,m)));
							vox_now.neighbors[5].push_back(((int) uv_5(1,m))*get<1>(image_size[m])+((int) uv_5(0,m)));
						}
						else {
							vox_now.pix_n.push_back(((int) uv(1,m))*get<1>(image_size[m])+((int) uv(0,m)));
							vox_now.neighbors[0].push_back(((int) uv_0(1,m))*get<1>(image_size[m])+((int) uv_0(0,m)));
							vox_now.neighbors[1].push_back(((int) uv_1(1,m))*get<1>(image_size[m])+((int) uv_1(0,m)));
							vox_now.neighbors[2].push_back(((int) uv_2(1,m))*get<1>(image_size[m])+((int) uv_2(0,m)));
							vox_now.neighbors[3].push_back(((int) uv_3(1,m))*get<1>(image_size[m])+((int) uv_3(0,m)));
							vox_now.neighbors[4].push_back(((int) uv_4(1,m))*get<1>(image_size[m])+((int) uv_4(0,m)));
							vox_now.neighbors[5].push_back(((int) uv_5(1,m))*get<1>(image_size[m])+((int) uv_5(0,m)));
						}
					}
					voxel_array.push_back(vox_now);
				}
			}
		}
	}
	
	return voxel_array;
}

inline int FocalGrid::IsVoxel(arma::Mat<double> uv) {

	int is_voxel = 0;

	int count = 0;

	for (int n=0; n<N_cam; n++) {
		if (uv(0,n)>=0 && uv(0,n)<(get<1>(image_size[n]))) {
			if (uv(1,n)>=0 && uv(1,n)<(get<0>(image_size[n]))) {
				count++;
			}
		}
	}

	if (count == N_cam) {
		is_voxel = 1;
	}

	return is_voxel;

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