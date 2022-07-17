### Neural Distributed Image Compression with Cross-Attention Feature Alignment

## Usage
### Clone
Clone this repository and enter the directory using the commands below:
```bash
git clone https://github.com/garjania/NDIC-CAFA.git
cd NDIC-CAFA/
```

### Requirements
`Python 3.7.3` is recommended.

Install the required packages with:
```bash
pip install -r requirements.txt
```
If you're having issues with installing PyTorch compatible with your CUDA version, we strongly suggest you refer to [related documentation page](https://pytorch.org/get-started/previous-versions/).

### Dataset
The datasets used for experiments are KITTI Stereo and Cityscape.

For KITTI Stereo and KITTI General you can download the necessary image pairs from [KITTI 2012](http://www.cvlibs.net/download.php?file=data_stereo_flow_multiview.zip) and [KITTI 2015](http://www.cvlibs.net/download.php?file=data_scene_flow_multiview.zip). After obtaining `data_stereo_flow_multiview.zip` and `data_scene_flow_multiview.zip`, run the following commands:
```bash
unzip data_stereo_flow_multiview.zip # KITTI 2012
mkdir data_stereo_flow_multiview
mv training data_stereo_flow_multiview
mv testing data_stereo_flow_multiview

unzip data_scene_flow_multiview.zip # KITTI 2015
mkdir data_scene_flow_multiview
mv training data_scene_flow_multiview
mv testing data_scene_flow_multiview
```

For Cityscape you can download the image pairs from [here](https://www.cityscapes-dataset.com/downloads/). After downloading `leftImg8bit_trainvaltest.zip` and `rightImg8bit_trainvaltest.zip`, run the following commands:
```bash
mkdir cityscape_dataset
unzip leftImg8bit_trainvaltest.zip
mv leftImg8bit cityscape_dataset
unzip rightImg8bit_trainvaltest.zip
mv rightImg8bit cityscape_dataset
```

### Getting Started
To use the code, please run:
```bash
python main.py
```
Please be aware that the initial console output (including the ongoing test results) might take a while to be printed. 

By default, this code uses the configurations in `configs/config.yaml`. You can either change this configuration file or create a new `yaml` file and use that for running. For this case, please run:
```bash
python main.py --config=path/to/new/config/file
```
### Configurations

- Dataset:
```yaml
dataset_name: 'KITTI_Stereo' # the name of the dataset. it can be either KITTI_Stereo, KITTI_General or Cityscape
dataset_path: '.' # for KITTI_Stereo or KITTI_General it's the txt files containing the real path of the images, and for Cityscape it's the path
                  # to the directory that contains leftImg8bit and rightImg8bit folders
resize: [128, 256]
```

`dataset_name` is the name of the dataset which will be used in the model. In case of using KITTI, `dataset_path` shows the path to `data_paths` directory that contains every image and its pair path, and for Cityscape it is the path to the directory that contains `leftImg8bit` and `rightImg8bit` folders. The `resize` value selects the width, and the height dimensions that each image will be resized to.

- Model:
```yaml
model: 'cross_attention' # acceptable values are: "bls17" for End-to-end Optimized Image Compression by Ballé, et al.,
                        #                        "bmshj18" for Variational image compression with a scale hyperprior by Ballé, et al.,
                        #                        "ndic_bls17" for NDIC model with Balle2017 baseline,
                        #                        "ndic_bmshj18" for NDIC model with Balle2018 baseline, and
                        #                        "cross_attention" for the Cross Attention model with Balle2017 baseline.
num_filters: 192 # number of filters used in the baseline model network
cuda: True
load_weight: False
weight_path: './pretrained_weights/model.pt' # weight path for loading
```

`model` selects the compression model. The accepted models for this parameter are `'bmshj18'` for [Variational image compression with a scale hyperprior](https://arxiv.org/abs/1802.01436), `'bls17'` for [End-to-end Optimized Image Compression](https://arxiv.org/abs/1611.01704), 
`'ndic_bls17'` and `'ndic_bmshj18'` for [Neural Distributed Image Compression using Common Information](https://arxiv.org/abs/2106.11723), and `'cross_attention'` for the proposed model. If `load_weight` is `True`, then in model initialization, the weight saved in `weight_path` is loaded to the model.



- Training
```yaml
train: True
epochs: 50000
train_batch_size: 1
lr: 0.0001
lambda: 0.00003 # the lambda value in rate-distortion equation
alpha: 0
beta: 0
distortion_loss: 'MS-SSIM' # can be MS-SSIM or MSE. selects the method by which the distortion is calculated during training
verbose_period: 50 # non-positive value indicates no verbose
```

For training, set `train` to be `True`. `lambda` shows the lambda value in the rate-distortion equation and `alpha` and `beta` correspond to the handles on the reconstruction of the correlated image and amount of common information extracted from the decoder-only side information, respectively. `distortion_loss` selects the distortion evaluating method. Its accepted values are MS-SSIM for the ms-ssim method or MSE for mean squared error.
`verbose_period: 50` indicates that every 50 epochs print the results of the validation dataset.

- Weight parameters
```yaml
save_weights: True
save_output_path: './outputs' # path where results and weights will be saved
experiment_name: 'cross_attention_MS-SSIM_lambda:3e-05''
```

If you wish to save the model weights after training, set `save_weights` `True`. `save_output_path` shows the directory path where the model weights are saved.
For the weights, in `save_output_path` a `weight` folder will be created, and the weights will be saved there with the name according to `experiment_name`. 

- Test:
```yaml
test: True
save_image: True
experiment_name: 'cross_attention_MS-SSIM_lambda:3e-05'
```

If you wish to test the model and save the results set `test` to `True`. If `save_image` is set to `True` then a `results` folder will be created, and the reconstructed images will be saved in `save_output_path/results` during testing, with the results named according to `experiment_name`.


### Inference 

In order to (only) carry out inference, please open `configs/config.yaml` and change the relevant lines as follows:

```yaml
resize: [128, 256] # we used this crop size for our inference
dataset_path: '.'
train: False
load_weight: True
test: True
save_output_path: './inference' 
save_image: True 
``` 

Based on the weight you chose, specify the weight path in `configs/config.yaml`:

```yaml
weight_path: './pretrained_weights/...'  # load a specified pre-trained weight
experiment_name: '...' # a handle for the saved results of the inference
```

Also, change the `model` parameter in `configs/config.yaml` accordingly.
For example, for the `cross_attention` weights, the parameter should be: 

```yaml
model: 'cross_attention'
```

After running the code using the command below, the results will be saved in `inference` folder.
```bash
python main.py
```

### Lambda Values
Here are some of the lambda values used for each dataset in order to obtain the results in the paper:

- Cityscape:
```
4.5e-05, 6e-05, 0.0001, 0.00016, 0.00022, 0.00032,  0.0004, 0.00044, 0.0005, 0.00062
```

- KITTI Stereo:
```
3e-05, 4.5e-05, 6e-05, 0.00011 , 0.00018, 0.00022, 0.0003
```

- KITTI General:
```
3e-05, 4.5e-05, 8e-05, 0.00011, 0.00016, 0.00022, 0.00032, 0.00042, 0.00046
```

### License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

### Authors/Contributors

* Ali Garjani* (github: [garjania](https://github.com/garjania))
* Nitish Mital* (github: [nitishmital](https://github.com/nitishmital))
* Ezgi Ozyilkan* (github: [ezgimez](https://github.com/ezgimez))
* Deniz Gunduz (github: [dgunduz](https://github.com/dgunduz))

*These authors contributed equally to this work.