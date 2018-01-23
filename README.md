# Hyperparameter optimization for neural networks

This repository contains code to experiment with dlib's recently released [global optimizer](http://dlib.net/optimization.html#global_function_search) for neural network hyperparameter optimization. 

## Prerequisites 

* **dlib**: Install dlib by cloning the [repository](https://github.com/davisking/dlib) and following the instructions there.
* **TF slim**: Clone the [TF models repository](https://github.com/tensorflow/models). Add slim to your `PYTHONPATH`:
  ```bash
  export PYTHONPATH=$PYTHONPATH:/path_to_your_folder/models/research/slim
  ```
* **Python packages:** Install all requirements via pip by
  ```bash
  pip install -r requirements.txt
  ``` 

Download the binary version of CIFAR-100 from [here](http://www.cs.toronto.edu/~kriz/cifar.html) or run
```bash
wget http://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz
```

## Usage

For running optimization over the three hyperparameters *depth_multiplier*, *weight_decay* and *dropout_keep_prob* with default settings, run
```bash
python optimize.py --data_dir <DATA_DIR> --out_dir <OUT_DIR>
```
