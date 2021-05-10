# Watershed evaluation

## Grid search to find the best parameters of watershed.

Once we pick the best model with largest F-score, waterhsed algorithm is run and using grid search to get the best parameter settings.

```
python grid_search_watershed.py <watershed_dir> <EPM_path> <label_path> <output_dir> <validation_mask> <auc-threshold> <input_results_dir>
```

## Plot the watershed evaluation results

```
python plot_eval_graph.py <input_results_dir> <input_results_dir> <output_file_path> <output_file_path_sen_1> <output_file_path_sen_2> <best_dynamic> <best_minimum_area>
```

## Optimum choices for different models in 30 epochs:

HED:

Optimum solution after grid search: dynamic minimum 0 and area minimum 300 -> F1: 0.245

Worst value after grid search: dynamic minimum 0 and area minimum 50 -> F1: 0.200

BDCN:

Optimum solution after grid search: dynamic minimum 1 and area minimum 100 -> F1: 0.244

Worst value after grid search: dynamic minimum 5 and area minimum 500 -> F1: 0.223

UNET:

Optimum solution after grid search: dynamic minimum 0 and area minimum 50 -> F1: 0.246

Worst value after grid search: dynamic minimum 3 and area minimum 500 -> F1: 0.215


## Evaluation benchmark score

| Pipeline        | parameters details and best values | F1 AUC AUC Threshold=0.5 |
| --------------- | ---------------------------------- | ------------------------ |
| HED + bin + WS  | dyn: 0 + area: 300                 |      0.245               |
| BDCN + bin + WS | dyn: 1 + area: 100                 |      0.244               |
| UNet + bin + WS | dyn: 0 + area: 50                  |      0.246               |


## Sensitivity results for HED + watershed

<img src="./eval_merge/dynamic_sensitivty_plot_HED.png"  width="400" height="400"> <img src="./eval_merge/minimum_area_sensitivty_plot_HED.png"  width="400" height="400">

## Sensitivity results for BDCN + watershed

<img src="./eval_merge/dynamic_sensitivty_plot_BDCN.png"  width="400" height="400"> <img src="./eval_merge/minimum_area_sensitivty_plot_BDCN.png"  width="400" height="400">

## Sensitivity results for UNET + watershed

<img src="./eval_merge/dynamic_sensitivty_plot_UNET.png"  width="400" height="400"> <img src="./eval_merge/minimum_area_sensitivty_plot_UNET.png"  width="400" height="400">
