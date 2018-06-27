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
#include "json.hpp"

using namespace std;

using json = nlohmann::json;

struct model_dat_layout {

	int N_components;
	vector<string> stl_list;
	vector<vector<int>> parent_list;
	vector<int> joint_type_list;
	vector<vector<double>> joint_param_parent;
	vector<vector<double>> joint_param_child;

};

int main ()
{

	//----------------------------------------------------------------------------------------------------
	// Drosophila model rigid wing
	//----------------------------------------------------------------------------------------------------
	int N_components = 5;

	// List the stl files of the model
	vector<string> stl_list = {"thorax.stl",
		"head.stl",
		"abdomen.stl",
		"wing_L.stl",
		"wing_R.stl"};

	// List with the kinematic tree structure
	vector<vector<int>> parent_list = {{0},
		{0,1},
		{0,2},
		{0,3},
		{0,4}};

	// List with the different joint types (6 DOF = 0, Ball joint = 1, Spherical surface joint = 2, 
	// Single axis of rotation joint = 3, Double axis of rotation joint = 4, Rigid joint = 5)
	vector<int> joint_type_list = {	0,
		1,
		4,
		2,
		2};

	vector<vector<double>> joint_param_parent = {{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
		{1.0, 0.0, 0.0, 0.0, 0.2058, 0.0, 0.0},
		{1.0, 0.0, 0.0, 0.0, -0.040, 0.0, -0.18},
		{1.0, 0.0, 0.0, 0.0, 0.0, 0.15, 0.0},
		{1.0, 0.0, 0.0, 0.0, 0.0, -0.15, 0.0}};

	vector<vector<double>> joint_param_child = {{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
		{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
		{0.9397, 0.0, -0.3420, 0.0, 0.0, 0.0, 0.0},
		{1.0, 0.0, 0.0, 0.0, 0.0, 0.1653, 0.0},
		{1.0, 0.0, 0.0, 0.0, 0.0, -0.1653, 0.0}};

	vector<vector<double>> bounding_box_config = {{0.9659, 0.0, 0.2588, 0.0, 0.075, 0.0, 0.0},
		{1.0, 0.0 ,0.0 ,0.0, 0.2058 ,0.0, 0.0},
		{0.9397, 0.0, -0.3420, 0.0, -0.040, 0.0, -0.18},
		{1.0, 0.0, 0.0, 0.0, 0.1, 0.0 ,0.0},
		{1.0, 0.0, 0.0, 0.0, 0.1, 0.0 ,0.0}};

	// Create json object:

	json config_file;

	config_file["N_components"] = N_components;
	config_file["stl_list"] = stl_list;
	config_file["parent_list"] = parent_list;
	config_file["joint_type_list"] = joint_type_list;
	config_file["joint_param_parent"] = joint_param_parent;
	config_file["joint_param_child"] = joint_param_child;
	config_file["bounding_box_config"] = bounding_box_config;

	cout << config_file.dump(2) << endl;

	// save json object

	chdir("/home/flyami/flight_tracker/models/drosophila_rigid_wing");

	ofstream outfile("drosophila_rigid_wing.json");
	outfile << config_file.dump(2) << endl;

	//----------------------------------------------------------------------------------------------------

	/*
	//----------------------------------------------------------------------------------------------------
	// Drosophila model segmented wing
	//----------------------------------------------------------------------------------------------------
	int N_components = 13;

	// List the stl files of the model
	vector<string> stl_list = {"thorax.stl",
							"head.stl",
							"abdomen.stl",
							"wing_sect_1_L.stl",
							"wing_sect_2_L.stl",
							"wing_sect_3_L.stl",
							"wing_sect_4_L.stl",
							"wing_sect_5_L.stl",
							"wing_sect_1_R.stl",
							"wing_sect_2_R.stl",
							"wing_sect_3_R.stl",
							"wing_sect_4_R.stl",
							"wing_sect_5_R.stl"};

	// List with the kinematic tree structure
	vector<vector<int>> parent_list = {{0},
							{0,1},
							{0,2},
							{0,3},
							{0,3,4},
							{0,3,4,5},
							{0,3,4,5,6},
							{0,3,4,5,6,7},
							{0,8},
							{0,8,9},
							{0,8,9,10},
							{0,8,9,10,11},
							{0,8,9,10,11,12}};

	// List with the different joint types (6 DOF = 0, Ball joint = 1, Spherical surface joint = 2, 
	// Single axis of rotation joint = 3, Double axis of rotation joint = 4, Rigid joint = 5)
	vector<int> joint_type_list = {0,
							1,
							4,
							2,
							1,
							5,
							5,
							5,
							2,
							1,
							5,
							5,
							5};

	vector<vector<double>> joint_param_parent = {{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
		{1.0, 0.0, 0.0, 0.0, 0.2058, 0.0, 0.0},
		{1.0, 0.0, 0.0, 0.0, -0.040, 0.0, -0.18},
		{1.0, 0.0, 0.0, 0.0, 0.0, 0.15, 0.0},
		{1.0, 0.0, 0.0, 0.0, 0.0, 0.15, 0.0},
		{1.0, 0.0, 0.0, 0.0, 0.0, 0.15, 0.0},
		{1.0, 0.0, 0.0, 0.0, 0.0, 0.15, 0.0},
		{1.0, 0.0, 0.0, 0.0, 0.0, 0.15, 0.0},
		{1.0, 0.0, 0.0, 0.0, 0.0, -0.15, 0.0},
		{1.0, 0.0, 0.0, 0.0, 0.0, -0.15, 0.0},
		{1.0, 0.0, 0.0, 0.0, 0.0, -0.15, 0.0},
		{1.0, 0.0, 0.0, 0.0, 0.0, -0.15, 0.0},
		{1.0, 0.0, 0.0, 0.0, 0.0, -0.15, 0.0}};

	vector<vector<double>> joint_param_child = {{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
		{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
		{-0.9397, 0.0, 0.3420, 0.0, 0.0, 0.0, 0.0},
		{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
		{1.0, 0.0, 0.0, 0.0, 0.042, 0.1649, 0.0},
		{1.0, 0.0, 0.0, 0.0, 0.0014, 0.1649, 0.0},
		{1.0, 0.0, 0.0, 0.0, 0.0014, 0.1649, 0.0},
		{1.0, 0.0, 0.0, 0.0, -0.0290, 0.1649, 0.0},
		{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
		{1.0, 0.0, 0.0, 0.0, 0.042, -0.1649, 0.0},
		{1.0, 0.0, 0.0, 0.0, 0.0014, -0.1649, 0.0},
		{1.0, 0.0, 0.0, 0.0, 0.0014, -0.1649, 0.0},
		{1.0, 0.0, 0.0, 0.0, -0.0290, -0.1649, 0.0}};

	// Create json object:

	json config_file;

	config_file["N_components"] = N_components;
	config_file["stl_list"] = stl_list;
	config_file["parent_list"] = parent_list;
	config_file["joint_type_list"] = joint_type_list;
	config_file["joint_param_parent"] = joint_param_parent;
	config_file["joint_param_child"] = joint_param_child;

	cout << config_file.dump(2) << endl;

	// save json object

	chdir("/home/flyami/flight_tracker/models/drosophila_seg_wing");

	ofstream outfile("drosophila_seg_wing.json");
	outfile << config_file.dump(2) << endl;

	//----------------------------------------------------------------------------------------------------
	*/
	

	return 0;
}