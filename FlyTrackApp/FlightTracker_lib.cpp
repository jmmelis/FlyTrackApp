#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include "flight_tracker_class.h"

namespace p = boost::python;
namespace np = boost::python::numpy;

BOOST_PYTHON_MODULE(FlightTracker_lib)
{
	Py_Initialize();
	np::initialize();
	p::class_<FLT>("FLT")
		.def("set_session_loc", &FLT::SetSessionLoc)
		.def("get_session_loc", &FLT::GetSessionLoc)
		.def("set_mov_name", &FLT::SetMovName)
		.def("get_mov_name", &FLT::GetMovName)
		.def("set_cam_name", &FLT::SetCamName)
		.def("get_cam_name", &FLT::GetCamName)
		.def("set_cal_loc", &FLT::SetCalLoc)
		.def("get_cal_loc", &FLT::GetCalLoc)
		.def("set_cal_name", &FLT::SetCalName)
		.def("get_cal_name", &FLT::GetCalName)
		.def("set_N_cam", &FLT::SetNrOfCam)
		.def("get_N_cam", &FLT::GetNrOfCam)
		.def("set_mov_nr", &FLT::SetMovNr)
		.def("get_mov_nr", &FLT::GetMovNr)
		.def("set_bckg_loc", &FLT::SetBckgLoc)
		.def("get_bckg_loc", &FLT::GetBckgLoc)
		.def("set_bckg_name", &FLT::SetBckgName)
		.def("get_bckg_name", &FLT::GetBckgName)
		.def("set_bckg_img_format", &FLT::SetBckgImageFormat)
		.def("get_bckg_img_format", &FLT::GetBckgImageFormat)
		.def("set_frame_name", &FLT::SetFrameName)
		.def("get_frame_name", &FLT::GetFrameName)
		.def("set_frame_img_format", &FLT::SetFrameImageFormat)
		.def("get_frame_img_format", &FLT::GetFrameImageFormat)
		.def("set_start_point", &FLT::SetStartPoint)
		.def("get_start_point", &FLT::GetStartPoint)
		.def("set_mid_point", &FLT::SetMidPoint)
		.def("get_mid_point", &FLT::GetMidPoint)
		.def("set_end_point", &FLT::SetEndPoint)
		.def("get_end_point", &FLT::GetEndPoint)
		.def("set_trigger_mode", &FLT::SetTriggerMode)
		.def("get_trigger_mode", &FLT::GetTriggerMode)
		.def("set_nx", &FLT::SetNx)
		.def("get_nx", &FLT::GetNx)
		.def("set_ny", &FLT::SetNy)
		.def("get_ny", &FLT::GetNy)
		.def("set_nz", &FLT::SetNz)
		.def("get_nz", &FLT::GetNz)
		.def("set_ds", &FLT::SetDs)
		.def("get_ds", &FLT::GetDs)
		.def("set_x0", &FLT::SetX0)
		.def("get_x0", &FLT::GetX0)
		.def("set_y0", &FLT::SetY0)
		.def("get_y0", &FLT::GetY0)
		.def("set_z0", &FLT::SetZ0)
		.def("get_z0", &FLT::GetZ0)
		.def("set_sol_loc", &FLT::SetSolLoc)
		.def("get_sol_loc", &FLT::GetSolLoc)
		.def("set_sol_name", &FLT::SetSolName)
		.def("get_sol_name", &FLT::GetSolName)
		.def("set_N_threads", &FLT::SetNumberOfThreads)
		.def("get_N_threads", &FLT::GetNumberOfThreads)
		.def("set_model_loc", &FLT::SetModelLoc)
		.def("get_model_loc", &FLT::GetModelLoc)
		.def("add_model", &FLT::AddModel)
		.def("clear_model_list", &FLT::ClearModelList)
		.def("set_session_par", &FLT::SetSessionParam)
		.def("init_frame_loader", &FLT::InitFrameLoader)
		.def("load_single_frame", &FLT::LoadSingleFrame)
		.def("load_frame_batch", &FLT::LoadFrameBatch)
		.def("return_single_frame", &FLT::ReturnSingleFrame)
		.def("return_image_size", &FLT::ReturnImageSize)
		.def("set_grid_param", &FLT::SetGridParameters)
		.def("construct_focal_grid", &FLT::ConstructGrid)
		.def("xyz_2_uv", &FLT::XYZ2UV)
		.def("drag_point_3d", &FLT::DragPoint3D)
		.def("load_model", &FLT::LoadModel)
		.def("set_seg_param", &FLT::SetImagSegmParam)
		.def("seg_single_frame", &FLT::SegmentSingleFrame)
		.def("seg_frame_batch", &FLT::SegmentFrameBatch)
		.def("return_seg_frame", &FLT::ReturnSegmentedFrame)
		.def("set_model_scale", &FLT::SetModelScale)
		.def("set_model_origin", &FLT::SetModelOrigin)
		.def("set_body_length", &FLT::SetBodyLength)
		.def("set_wing_length", &FLT::SetWingLength)
		.def("set_model_init_state", &FLT::SetInitState)
		.def("return_transform_mats", &FLT::ReturnTransformationMatrices)
		.def("set_weight_xsi", &FLT::SetWeightXsi)
		.def("set_weight_theta", &FLT::SetWeightTheta)
		.def("set_weight_length", &FLT::SetWeightLength)
		.def("set_weight_volume", &FLT::SetWeightVolume)
		.def("set_cone_angle", &FLT::SetConeAngle)
		.def("set_cone_height", &FLT::SetConeHeight)
		.def("single_frame_2_pcl", &FLT::ProjectSingleFrame2PCL)
		.def("return_segment_pcl", &FLT::ReturnSingleSegmentPCL)
		.def("find_initial_state", &FLT::FindInitialState)
		.def("return_blr_pcl", &FLT::ReturnBLRPCL)
		.def("return_blr_bboxes", &FLT::ReturnBBoxBLR)
		.def("return_m_body", &FLT::ReturnMBody)
		.def("return_m_wing_L", &FLT::ReturnMWingL)
		.def("return_m_wing_R", &FLT::ReturnMWingR)
		.def("return_bboxes", &FLT::ReturnBBoxSingleFrame)
		.def("return_parent_structure", &FLT::ReturnParentStructure)
		.def("return_model_state", &FLT::ReturnModelState)
		.def("return_stl_list", &FLT::ReturnSTLFileList)
		.def("return_model_scale", &FLT::ReturnModelScale)
		.def("return_projected_model_pcl", &FLT::ReturnProjectedModelPCL)
		.def("return_projected_frames", &FLT::ReturnProjectedFrame)
		.def("icp_on_single_frame", &FLT::PerformICPSingleFrame);
}