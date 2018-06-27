#include "model_class.h"

#include "model.h"
#include "frames.h"

#include <string>
#include <stdint.h>
#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <vector>
#include <algorithm>
#include <dirent.h>
#include <armadillo>
#include <pcl/io/pcd_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/point_types.h>
#include <pcl/console/parse.h>
#include <pcl/common/transforms.h>
#include <pcl/PCLPointCloud2.h>
#include "json.hpp"

//#include "models/drosophila_melanogaster/drosophila_model.h"
//#include "models/drosophila_melanogaster_simple/drosophila_model.h"

using namespace std;

using json = nlohmann::json;

//using namespace model_constants;

struct model_dat_layout {

	int N_components;
	vector<string> stl_list;
	vector<vector<int>> parent_list;
	vector<int> joint_type_list;
	vector<vector<double>> joint_param_parent;
	vector<vector<double>> joint_param_child;

};

ModelClass::ModelClass() {
	// empty
}

bool ModelClass::SaveCurrentModel(session_param &ses_par, model &mod, string file_name) {

	bool model_saved = true;

	try {

		FILE *outfile;

		string out_file_string = ses_par.sol_loc + "/model/" + file_name + ".dat";

		const char* out_file_name = out_file_string.c_str();

		outfile = fopen(out_file_name, "w");

		if (outfile == NULL) {
			cout << "could not create model file" << endl;
			model_saved = false;
		}
		else {

			fwrite(&mod, sizeof(struct model), 1, outfile);

			if (fwrite != 0) {
				cout << "model saved" << endl;
			}
			else {
				cout << "could not save the model" << endl;
				model_saved = false;
			}
		}
	}
	catch (...) {
		model_saved = false;
	}

	return model_saved;

}

bool ModelClass::LoadModel(session_param &ses_par, model &mod, int model_ind) {

	bool model_loaded = true;

	try {

		// Change directory:
		string dir_name = ses_par.model_loc + "/" + ses_par.model_folders[model_ind];
		//cout << "dir name: " + dir_name << endl;
		chdir(dir_name.c_str());

		// Load JSON file:
		string file_name = ses_par.model_file_names[model_ind] + ".json";
		//cout << "file name: " + file_name << endl;
		ifstream in_file(file_name.c_str());
		json loaded_file;
		in_file >> loaded_file;

		// Print content:
		//cout << "content: " + file_name << endl;
		//cout << loaded_file.dump(2) << endl;

		// Copy content to struct:
		//cout << loaded_file["N_components"] << endl;
		//cout << loaded_file["stl_list"] << endl;
		//cout << loaded_file["parent_list"] << endl;
		//cout << loaded_file["joint_type_list"] << endl;
		//cout << loaded_file["joint_param_parent"] << endl;
		//cout << loaded_file["joint_param_child"] << endl;

		mod.N_parts = loaded_file["N_components"];
		mod.stl_list.clear();
		mod.parts.clear();
		mod.parents.clear();
		mod.joint_param_parent.clear();
		mod.joint_param_child.clear();
		mod.bounding_box_config.clear();

		vector<string> stl_list = loaded_file["stl_list"];
		vector<vector<int>> parent_list = loaded_file["parent_list"];
		vector<int> joint_type_list = loaded_file["joint_type_list"];
		vector<vector<double>> joint_param_parent = loaded_file["joint_param_parent"];
		vector<vector<double>> joint_param_child = loaded_file["joint_param_child"];
		vector<vector<double>> bounding_box_config = loaded_file["bounding_box_config"];


		for (int i=0; i<mod.N_parts; i++) {
			mod.stl_list.push_back(stl_list[i]);
			mod.parts.push_back(ModelClass::LoadSTLFile(stl_list[i], ses_par.model_loc + "/" + ses_par.model_folders[model_ind]));
			mod.parents.push_back(parent_list[i]);
			arma::Col<double> joint_parent = {joint_param_parent[i][0], joint_param_parent[i][1], joint_param_parent[i][2], joint_param_parent[i][3], 
				joint_param_parent[i][4], joint_param_parent[i][5], joint_param_parent[i][6]};
			mod.joint_param_parent.push_back(joint_parent);
			arma::Col<double> joint_child = {joint_param_child[i][0], joint_param_child[i][1], joint_param_child[i][2], joint_param_child[i][3], 
				joint_param_child[i][4], joint_param_child[i][5], joint_param_child[i][6]};
			mod.joint_param_child.push_back(joint_child);
			arma::Col<double> bounding_box_cfg = {bounding_box_config[i][0],bounding_box_config[i][1],bounding_box_config[i][2],bounding_box_config[i][3],
				bounding_box_config[i][4],bounding_box_config[i][5],bounding_box_config[i][6]};
			mod.bounding_box_config.push_back(bounding_box_cfg);
			arma::Col<double> temp_state = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
			mod.state.push_back(temp_state);
			mod.scale.push_back(1.0);
			mod.alpha.push_back(1.0);
		}

		 ModelClass::SetStartState(mod);
	}
	catch (...) {
		model_loaded = false;
	}

	return model_loaded;
}

pcl::PolygonMesh ModelClass::LoadSTLFile(string FileName, string FileLoc) {

	const string stl_file_name = FileLoc + "/" + FileName;

	pcl::PolygonMesh stl_mesh;

	pcl::io::loadPolygonFileSTL(stl_file_name,stl_mesh);

	return stl_mesh;
}

vector<string> ModelClass::ReturnSTLNames(model &mod) {

	vector<string> stl_file_list;

	for (int i=0; i<mod.N_parts; i++) {
		stl_file_list.push_back(mod.stl_list[i]);
	}

	return stl_file_list;
}

void ModelClass::SetScale(model &mod, arma::Col<double> scale_vec) {

	mod.scale.clear();

	for (int i=0; i<scale_vec.n_rows; i++) {
		mod.scale.push_back(scale_vec(i));
	}

}

void ModelClass::SetOrigin(model &mod, arma::Col<double> origin_loc) {
	mod.origin_loc = origin_loc;
}

void ModelClass::SetBodyLength(model &mod, double body_length) {
	mod.body_length = body_length;
	cout << body_length << endl;
}

void ModelClass::SetWingLength(model &mod, double wing_length) {
	mod.wing_length = wing_length;
	cout << wing_length << endl;
}

void ModelClass::SetStartState(model &mod) {

	mod.state.clear();
	mod.M_vec.clear();

	int N_seg = mod.N_parts;

	for (int i=0; i<N_seg; i++) {

		int level = mod.parents[i].size();

		if (level==1) {
			arma::Col<double> temp_state;
			temp_state = mod.joint_param_child[i];
			mod.state.push_back(temp_state);
			mod.M_vec.push_back(ModelClass::GetM(temp_state));
		}
		else {
			int parent_ind = mod.parents[i][level-1];
			arma::Col<double> temp_state;
			temp_state = mod.joint_param_child[i];
			temp_state(4) = mod.joint_param_child[i](4)*mod.scale[i]+mod.joint_param_parent[i](4)*mod.scale[parent_ind];
			temp_state(5) = mod.joint_param_child[i](5)*mod.scale[i]+mod.joint_param_parent[i](5)*mod.scale[parent_ind];
			temp_state(6) = mod.joint_param_child[i](6)*mod.scale[i]+mod.joint_param_parent[i](6)*mod.scale[parent_ind];
			mod.state.push_back(temp_state);
			mod.M_vec.push_back(ModelClass::GetM(temp_state));
		}

	}

}

/*
void ModelClass::SetInitState(model &mod, frames &frame_in) {

	mod.state.clear();

	arma::Mat<double> M_in;

	for (int i=0; i<5; i++) {
		if (i<3) {
			M_in = frame_in.M_init_single_frame[0];
			arma::Col<double> temp_state = ModelClass::GetStateVec(M_in);
			mod.state.push_back(temp_state);
		}
		else if (i==3) {
			M_in = frame_in.M_init_single_frame[1].submat(0,0,3,3);
			arma::Col<double> temp_state = ModelClass::GetStateVec(M_in);
			mod.state.push_back(temp_state);
		}
		else if (i==4) {
			M_in = frame_in.M_init_single_frame[2].submat(0,0,3,3);
			arma::Col<double> temp_state = ModelClass::GetStateVec(M_in);
			mod.state.push_back(temp_state);
		}
	}

}
*/

void ModelClass::SetInitState(model &mod, frames &frame_in) {

	//mod.state.clear();

	mod.M_vec.clear();

	int N_seg = mod.N_parts;

	for (int i=0; i<N_seg; i++) {

		if (i<3) {

			int level = mod.parents[i].size();

			if (level==1) {
				arma::Mat<double> M_body = frame_in.M_init_single_frame[0];
				arma::Mat<double> R_body = M_body.submat(0,0,2,2);
				arma::Col<double> T_body = {M_body(0,3), M_body(1,3), M_body(2,3)};

				double body_scale = mod.scale[i];

				arma::Col<double> bbox_config = mod.bounding_box_config[i];
				arma::Mat<double> M_bbox = ModelClass::GetM(bbox_config);
				arma::Mat<double> R_bbox = M_bbox.submat(0,0,2,2);
				arma::Col<double> T_bbox = {M_bbox(0,3)*body_scale, M_bbox(1,3)*body_scale, M_bbox(2,3)*body_scale};

				arma::Mat<double> R_out = R_body*R_bbox;
				arma::Col<double> T_out = R_body*T_bbox+T_body;
				arma::Mat<double> M_out = {{R_out(0,0), R_out(0,1), R_out(0,2), T_out(0)},
					{R_out(1,0), R_out(1,1), R_out(1,2), T_out(1)},
					{R_out(2,0), R_out(2,1), R_out(2,2), T_out(2)},
					{0.0, 0.0 ,0.0 ,1.0}};
				mod.M_vec.push_back(M_out);
			}
			else {
				arma::Mat<double> M_body = mod.M_vec[0];
				arma::Mat<double> R_body = M_body.submat(0,0,2,2);
				arma::Col<double> T_body = {M_body(0,3), M_body(1,3), M_body(2,3)};

				double body_scale = mod.scale[i];

				arma::Col<double> bbox_config = mod.bounding_box_config[i];
				arma::Mat<double> M_bbox = ModelClass::GetM(bbox_config);
				arma::Mat<double> R_bbox = M_bbox.submat(0,0,2,2);
				arma::Col<double> T_bbox = {M_bbox(0,3)*body_scale, M_bbox(1,3)*body_scale, M_bbox(2,3)*body_scale};

				arma::Mat<double> R_out = R_body*R_bbox;
				arma::Col<double> T_out = R_body*T_bbox+T_body;
				arma::Mat<double> M_out = {{R_out(0,0), R_out(0,1), R_out(0,2), T_out(0)},
					{R_out(1,0), R_out(1,1), R_out(1,2), T_out(1)},
					{R_out(2,0), R_out(2,1), R_out(2,2), T_out(2)},
					{0.0, 0.0 ,0.0 ,1.0}};
				mod.M_vec.push_back(M_out);
			}
		}
		else if (i==3) {
			arma::Mat<double> M_body = frame_in.M_init_single_frame[1];
			arma::Mat<double> R_body = M_body.submat(0,0,2,2);
			arma::Col<double> T_body = {M_body(0,3), M_body(1,3), M_body(2,3)};

			double wing_scale = mod.scale[i];

			arma::Col<double> bbox_config = mod.bounding_box_config[i];
			arma::Mat<double> M_bbox = ModelClass::GetM(bbox_config);
			arma::Mat<double> R_bbox = M_bbox.submat(0,0,2,2);
			arma::Col<double> T_bbox = {M_bbox(0,3)*wing_scale, M_bbox(1,3)*wing_scale, M_bbox(2,3)*wing_scale};

			arma::Mat<double> R_out = R_body*R_bbox;
			arma::Col<double> T_out = R_body*T_bbox+T_body;
			arma::Mat<double> M_out = {{R_out(0,0), R_out(0,1), R_out(0,2), T_out(0)},
				{R_out(1,0), R_out(1,1), R_out(1,2), T_out(1)},
				{R_out(2,0), R_out(2,1), R_out(2,2), T_out(2)},
				{0.0, 0.0 ,0.0 ,1.0}};
			mod.M_vec.push_back(M_out);
		}
		else if (i==4) {
			arma::Mat<double> M_body = frame_in.M_init_single_frame[2];
			arma::Mat<double> R_body = M_body.submat(0,0,2,2);
			arma::Col<double> T_body = {M_body(0,3), M_body(1,3), M_body(2,3)};

			double wing_scale = mod.scale[i];

			arma::Col<double> bbox_config = mod.bounding_box_config[i];
			arma::Mat<double> M_bbox = ModelClass::GetM(bbox_config);
			arma::Mat<double> R_bbox = M_bbox.submat(0,0,2,2);
			arma::Col<double> T_bbox = {M_bbox(0,3)*wing_scale, M_bbox(1,3)*wing_scale, M_bbox(2,3)*wing_scale};

			arma::Mat<double> R_out = R_body*R_bbox;
			arma::Col<double> T_out = R_body*T_bbox+T_body;
			arma::Mat<double> M_out = {{R_out(0,0), R_out(0,1), R_out(0,2), T_out(0)},
				{R_out(1,0), R_out(1,1), R_out(1,2), T_out(1)},
				{R_out(2,0), R_out(2,1), R_out(2,2), T_out(2)},
				{0.0, 0.0 ,0.0 ,1.0}};
			mod.M_vec.push_back(M_out);
		}

	}

}

vector<arma::Mat<double>> ModelClass::ReturnTransformMatrices(model &mod) {

	vector<arma::Mat<double>> M_vec;

	int N_seg = mod.N_parts;

	for (int i=0; i<N_seg; i++) {

		/*
		arma::Col<double> q_vec = {mod.state[i](0),mod.state[i](1),mod.state[i](2),mod.state[i](3)};
		q_vec = q_vec/arma::norm(q_vec,2);

		double q0 = q_vec (0);
		double q1 = q_vec (1);
		double q2 = q_vec (2);
		double q3 = q_vec (3);
		double tx = mod.state[i](4);
		double ty = mod.state[i](5);
		double tz = mod.state[i](6);

		arma::Mat<double> M = {{2.0*pow(q0,2.0)-1.0+2.0*pow(q1,2.0), 2.0*q1*q2-2.0*q0*q3, 2.0*q1*q3+2.0*q0*q2, tx},
			{2.0*q1*q2+2.0*q0*q3, 2.0*pow(q0,2.0)-1.0+2.0*pow(q2,2.0), 2.0*q2*q3-2.0*q0*q1, ty},
			{2.0*q1*q3-2.0*q0*q2, 2.0*q2*q3+2.0*q0*q1, 2.0*pow(q0,2.0)-1.0+2.0*pow(q3,2.0), tz},
			{0.0, 0.0, 0.0, 1.0}};

		//arma::Mat<double> M = {{2.0*pow(q0,2)-1.0+2.0*pow(q1,2), 2.0*q1*q2+2.0*q0*q3, 2.0*q1*q3-2.0*q0*q2, tx},
		//	{2.0*q1*q2-2.0*q0*q3, 2.0*pow(q0,2)-1.0+2.0*pow(q2,2), 2.0*q2*q3+2.0*q0*q1, ty},
		//	{2.0*q1*q3+2.0*q0*q2, 2.0*q2*q3-2.0*q0*q1, 2.0*pow(q0,2)-1.0+2.0*pow(q3,2), tz},
		//	{0.0, 0.0, 0.0, 1.0}};
		*/

		//arma::Mat<double> M = ModelClass::GetM(mod.state[i]);

		M_vec.push_back(mod.M_vec[i]);

	}

	return M_vec;
}

arma::Mat<double> ModelClass::GetM(arma::Col<double> state_in) {

	arma::Col<double> q_vec = {state_in(0),state_in(1),state_in(2),state_in(3)};
	//q_vec = q_vec/arma::norm(q_vec,2);
	q_vec = q_vec/sqrt(pow(q_vec(0),2)+pow(q_vec(1),2)+pow(q_vec(2),2)+pow(q_vec(3),2));

	double q0 = q_vec(0);
	double q1 = q_vec(1);
	double q2 = q_vec(2);
	double q3 = q_vec(3);
	double tx = state_in(4);
	double ty = state_in(5);
	double tz = state_in(6);

	//arma::Mat<double> M = {{2.0*pow(q0,2.0)-1.0+2.0*pow(q1,2.0), 2.0*q1*q2+2.0*q0*q3, 2.0*q1*q3-2.0*q0*q2, tx},
	//	{2.0*q1*q2-2.0*q0*q3, 2.0*pow(q0,2.0)-1.0+2.0*pow(q2,2.0), 2.0*q2*q3+2.0*q0*q1, ty},
	//	{2.0*q1*q3+2.0*q0*q2, 2.0*q2*q3-2.0*q0*q1, 2.0*pow(q0,2.0)-1.0+2.0*pow(q3,2.0), tz},
	//	{0.0, 0.0, 0.0, 1.0}};

	arma::Mat<double> M = {{2.0*pow(q0,2.0)-1.0+2.0*pow(q1,2.0), 2.0*q1*q2-2.0*q0*q3, 2.0*q1*q3+2.0*q0*q2, tx},
		{2.0*q1*q2+2.0*q0*q3, 2.0*pow(q0,2.0)-1.0+2.0*pow(q2,2.0), 2.0*q2*q3-2.0*q0*q1, ty},
		{2.0*q1*q3-2.0*q0*q2, 2.0*q2*q3+2.0*q0*q1, 2.0*pow(q0,2.0)-1.0+2.0*pow(q3,2.0), tz},
		{0.0, 0.0, 0.0, 1.0}};

	return M;

}

arma::Col<double> ModelClass::GetStateVec(arma::Mat<double> M_in) {

	arma::Col<double> state_out;

	state_out.zeros(7);
	//state_out(0) = 1.0;

	//if (M_in(3,3) == 1.0) {
	arma::Col<double> q_vec(4);
	q_vec(0) = 0.5*sqrt(M_in(0,0)+M_in(1,1)+M_in(2,2)+1.0);
	q_vec(1) = (M_in(1,2)-M_in(2,1))/(4.0*q_vec(0));
	q_vec(2) = (M_in(2,0)-M_in(0,2))/(4.0*q_vec(0));
	q_vec(3) = (M_in(0,1)-M_in(1,0))/(4.0*q_vec(0));
	//q_vec(1) = (M_in(2,1)-M_in(1,2))/(4.0*q_vec(0));
	//q_vec(2) = (M_in(0,2)-M_in(2,0))/(4.0*q_vec(0));
	//q_vec(3) = (M_in(1,0)-M_in(0,1))/(4.0*q_vec(0));
	q_vec = q_vec/sqrt(pow(q_vec(0),2)+pow(q_vec(1),2)+pow(q_vec(2),2)+pow(q_vec(3),2));
	state_out(0) = q_vec(0);
	state_out(1) = q_vec(1);
	state_out(2) = q_vec(2);
	state_out(3) = q_vec(3);
	state_out(4) = M_in(0,3);
	state_out(5) = M_in(1,3);
	state_out(6) = M_in(2,3);
	//}

	return state_out; 
}

arma::Col<double> ModelClass::MultiplyStates(arma::Col<double> state_A, arma::Col<double> state_B) {

	arma::Col<double> q_A = {state_A(0),state_A(1),state_A(2),state_A(3)};
	q_A = q_A/arma::norm(q_A,2);
	arma::Col<double> t_A = {state_A(4), state_A(5), state_A(6)};

	arma::Col<double> q_B = {state_B(0),state_B(1),state_B(2),state_B(3)};
	q_B = q_B/arma::norm(q_B,2);
	arma::Col<double> t_B = {state_B(4), state_B(5), state_B(6)};

	arma::Mat<double> Q_A = {{q_A(0), -q_A(1), -q_A(2), -q_A(3)},
							 {q_A(1),  q_A(0), -q_A(3),  q_A(2)},
							 {q_A(2),  q_A(3),  q_A(0), -q_A(1)},
							 {q_A(3), -q_A(2),  q_A(1),  q_A(0)}};

	arma::Col<double> q_C = Q_A*q_B;
	q_C = q_C/arma::norm(q_C,2);

	arma::Mat<double> R_A = ModelClass::GetM(state_A);

	arma::Col<double> t_C = t_A+R_A.submat(0,0,2,2)*t_B;

	arma::Col<double> state_out = {q_C(0), q_C(1), q_C(2), q_C(3), t_C(0), t_C(1), t_C(2)};

	return state_out;

}