import pipeline as pipeline
import argparse
import logging

logging.basicConfig(level=logging.DEBUG)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Point cloud processing.")

    # Positional argument
    parser.add_argument("input_folder", type=str, help="Path to the input folder containing a set of .ply or .pcd files")

    # Optional arguments
    parser.add_argument("-o", "--output_folder", type=str, help="Path to the output folder where the results will be exported", default=None)
    parser.add_argument("-param_registration_fitness_threshold", "--param_registration_fitness_threshold",
                         type=float, default=0.7, help="Fitness ratio for acceptable overlap between two registered point clouds (default: 0.7)")
    parser.add_argument("-param_registration_use_simple_coarse", "--param_registration_use_simple_coarse",
                         type=int, default=0, help="Use simple coarse registration, i.e. point-to-point (default: 0)")
    parser.add_argument("-param_outlier_removal_use_per_point_cloud_checking", "--param_outlier_removal_use_per_point_cloud_checking",
                         type=int, default=1, help="Use point cloud-based checking for outlier removal (default: 1)")
    parser.add_argument("-param_outlier_removal_use_visibility_confidence", "--param_outlier_removal_use_visibility_confidence",
                         type=int, default=1, help="Use visibility confidence for outlier removal (default: 1)")
    parser.add_argument("-param_export_intermediate_results", "--param_export_intermediate_results",
                         type=int, default=1, help="Export intermediate results (default: 1)")


    args = parser.parse_args()

    print("-------------------------------------------------------------------------------------------------------------------------")
    print("Arguments received:")
    print(f"  Input folder                                              : {args.input_folder}")
    print(f"  Output folder                                             : {args.output_folder}")
    print(f"  param_registration_fitness_threshold                      : {args.param_registration_fitness_threshold}")
    print(f"  param_registration_use_simple_coarse                      : {args.param_registration_use_simple_coarse}")
    print(f"  param_outlier_removal_use_per_point_cloud_checking        : {args.param_outlier_removal_use_per_point_cloud_checking}")
    print(f"  param_outlier_removal_use_visibility_confidence           : {args.param_outlier_removal_use_visibility_confidence}")
    print(f"  param_export_intermediate_results                         : {args.param_export_intermediate_results}")
    print("-------------------------------------------------------------------------------------------------------------------------")
    
    processing_pipeline = pipeline.ProcessingPipeline()
    processing_pipeline.param_registration_fitness_threshold = float(args.param_registration_fitness_threshold)
    processing_pipeline.param_registration_use_simple_coarse = args.param_registration_use_simple_coarse != 0
    processing_pipeline.param_outlier_removal_use_per_point_cloud_checking = args.param_outlier_removal_use_per_point_cloud_checking != 0
    processing_pipeline.param_outlier_removal_use_visibility_confidence = args.param_outlier_removal_use_visibility_confidence != 0
    processing_pipeline.param_export_intermediate_results = args.param_export_intermediate_results != 0
    processing_pipeline.prepare()

    processing_data = pipeline.ProcessingData()
    processing_data.output_folder_path = args.output_folder
    processing_data.load_from_folder(folder_path=args.input_folder)
    print("-------------------------------------------------------------------------------------------------------------------------")
    print(f"Processing {args.input_folder} and writing results in {processing_data.output_folder_path}")
    print("-------------------------------------------------------------------------------------------------------------------------")

    processing_pipeline.full_pipeline(processing_data)