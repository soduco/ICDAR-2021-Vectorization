
# BDCN pretrain
python grid_search_watershed.py --reconstruction_image_path ../benchmark/best_models/best_models_prediction/BDCN_pretrain_best_val.png --input_results_dir ./output/BDCN_pretrain/eval_ws/

# BDCN scratch
python grid_search_watershed.py --reconstruction_image_path ../benchmark/best_models/best_models_prediction/BDCN_scratch_best_val.png --input_results_dir ./output/BDCN_scratch/eval_ws/

# HED pretrain
python grid_search_watershed.py --reconstruction_image_path ../benchmark/best_models/best_models_prediction/HED_pretrain_best_val.png --input_results_dir ./output/HED_pretrain/eval_ws/

# HED scratch
python grid_search_watershed.py --reconstruction_image_path ../benchmark/best_models/best_models_prediction/HED_scratch_best_val.png --input_results_dir ./output/HED_scratch/eval_ws/

# UNET
python grid_search_watershed.py --reconstruction_image_path ../benchmark/best_models/best_models_prediction/UNET_scratch_best_val.png --input_results_dir ./output/UNET_scratch/eval_ws/
