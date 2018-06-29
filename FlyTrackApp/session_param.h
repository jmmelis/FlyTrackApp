#ifndef SESSION_PARAM_H
#define SESSION_PARAM_H

#include <string>
#include <stdint.h>
#include <vector>

using namespace std;

struct session_param {
	int N_cam;
	int mov_nr;
	int start_point;
	int mid_point;
	int end_point;
	string trigger_mode;
	vector<int> chrono_frame_nr;
	string session_loc;
	string mov_name;
	string cam_name;
	string cal_loc;
	string cal_name;
	string bckg_loc;
	string bckg_name;
	string bckg_img_format;
	string frame_name;
	string frame_img_format;
	string sol_loc;
	string sol_name;
	string model_loc;
	vector<string> model_folders;
	vector<string> model_file_names;
};
#endif