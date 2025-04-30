# Uncertainty Calibration in Deep Neural Networks: Bridging Classification and Time-Series Prediction


## Prerequisites

- Python 3
- CPU or NVIDIA GPU + CUDA

## Classification Calibration 

### Datasets
CIFAR-10 and CIFAR-100 datasets will be downloaded automatically.

### Experiments

You can train and evaluate a model with meta-calibration using the following commands:
```
python train.py --dataset cifar10 --model resnet18 --loss class_label --save-path Models/ --exp_name rn18_c10_meta_calibration --meta_calibration non_uniform_label_smoothing

python evaluate.py --dataset cifar10 --model resnet18 --save-path Models/ --saved_model_name rn18_c10_meta_calibration_best.model --exp_name rn18_c10_meta_calibration -log
```

## Time-Series Prediction Calibration 

"Replace file 'variational_estimator', which is highly likely located in:"Anaconda\envs\tensorflow-gpu2024\Lib\site-packages\blitz\utils"

run  LSTM_calibration（sin）.py

