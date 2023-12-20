## Multi_pose face image generation by modifying latent space<br>


Abstract: *Synthesis and reconstruction of 3D human head has gained increasing interest in computer vision and computer graphics recently. Existing state-of-the-art 3D generative adversarial networks (GANs) for 3D human head synthesis are either limited to near-frontal views or hard to preserve 3D consistency in large view angles. We propose a new method, the first 3D-aware generative model that enables high-quality view-consistent image synthesis of full heads in 360° with diverse appearance and detailed geometry using only in-the-wild unstructured images for training. At its core, we lift up the representation power of recent 3D GANs and bridge the data alignment gap when training from in-the-wild images with widely distributed views. Specifically, we propose a novel two-stage self-adaptive image alignment for robust 3D GAN training.


## Requirements

* We recommend Linux for performance and compatibility reasons.
* 1&ndash;8 high-end NVIDIA GPUs. We have done all testing and development using V100, RTX3090, and A100 GPUs.
* 64-bit Python 3.8 and PyTorch 1.11.0 (or later). See https://pytorch.org for PyTorch install instructions.
* CUDA toolkit 11.3 or later.  
* Python libraries: see [environment.yml](./environment.yml) for exact library dependencies.  You can use the following commands with Miniconda3 to create and activate your Python environment:
  - `cd multi_pose`
  - `conda env create -f environment.yml`
  - `conda activate multi_pose`


## Getting started

Download the whole `models` folder from [link](https://drive.google.com/drive/folders/1m517-F1NCTGA159dePs5R5qj02svtX1_?usp=sharing) and put it under the root dir.

Pre-trained networks are stored as `*.pkl` files that can be referenced using local filenames.


## Generating results

```.bash
# Generate videos using pre-trained model

python gen_videos.py --network models/easy-khair-180-gpc0.8-trans10-025000.pkl \
--seeds 0-3 --grid 2x2 --outdir=out --cfg Head --trunc 0.7

```

```.bash
# Generate images and shapes (as .mrc files) using pre-trained model

python gen_samples.py --outdir=out --trunc=0.7 --shapes=true --seeds=0-3 \
    --network models/easy-khair-180-gpc0.8-trans10-025000.pkl
```

## Applications
```.bash
# Generate full head reconstruction from a single RGB image.
# Please refer to ./gen_pti_script.sh
# For this application we need to specify dataset folder instead of zip files.
# Segmentation files are not necessary for PTI inversion.

./gen_pti_script.sh
```

```.bash
# Generate full head interpolation from two seeds.
# Please refer to ./gen_interpolation.py for the implementation

python gen_interpolation.py --network models/easy-khair-180-gpc0.8-trans10-025000.pkl\
        --trunc 0.7 --outdir interpolation_out
```



## Using networks from Python

You can use pre-trained networks in your own Python code as follows:

```.python
with open('*.pkl', 'rb') as f:
    G = pickle.load(f)['G_ema'].cuda()  # torch.nn.Module
z = torch.randn([1, G.z_dim]).cuda()    # latent codes
c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1) # camera parameters
img = G(z, c)['image']                           # NCHW, float32, dynamic range [-1, +1], no truncation
mask = G(z, c)['image_mask']                    # NHW, int8, [0,255]
```

The above code requires `torch_utils` and `dnnlib` to be accessible via `PYTHONPATH`. It does not need source code for the networks themselves &mdash; their class definitions are loaded from the pickle via `torch_utils.persistence`.

The pickle contains three networks. `'G'` and `'D'` are instantaneous snapshots taken during training, and `'G_ema'` represents a moving average of the generator weights over several training steps. The networks are regular instances of `torch.nn.Module`, with all of their parameters and buffers placed on the CPU at import and gradient computation disabled by default.


## Datasets format
For training purpose, we can use either zip files or normal folder for image dataset and segmentation dataset. For PTI, we need to use folder.

To compress dataset folder to zip file, we can use [dataset_tool_seg](./dataset_tool_seg.py). 

For example:
```.bash
python dataset_tool_seg.py --img_source dataset/testdata_img --seg_source  dataset/testdata_seg --img_dest dataset/testdata_img.zip --seg_dest dataset/testdata_seg.zip --resolution 512x512
```


## Training

Examples of training using `train.py`:

```
# Train with StyleGAN2 backbone from scratch with raw neural rendering resolution=64, using 8 GPUs.
# with segmentation mask, trigrid_depth@3, self-adaptive camera pose loss regularizer@10

python train.py --outdir training-runs  --img_data dataset/testdata_img.zip --seg_data dataset/testdata_seg.zip --cfg=ffhq --batch=32 --gpus 8\\
--gamma=1 --gamma_seg=1 --gen_pose_cond=True --mirror=1 --use_torgb_raw=1 --decoder_activation="none" --disc_module MaskDualDiscriminatorV2\\
--bcg_reg_prob 0.2 --triplane_depth 3 --density_noise_fade_kimg 200 --density_reg 0 --min_yaw 0 --max_yaw 180 --back_repeat 4 --trans_reg 10 --gpc_reg_prob 0.7


# Second stage finetuning to 128 neural rendering resolution (optional).

python train.py --outdir results --img_data dataset/testdata_img.zip --seg_data dataset/testdata_seg.zip --cfg=ffhq --batch=32 --gpus 8\\
--resume=~/training-runs/experiment_dir/network-snapshot-025000.pkl\\
--gamma=1 --gamma_seg=1 --gen_pose_cond=True --mirror=1 --use_torgb_raw=1 --decoder_activation="none" --disc_module MaskDualDiscriminatorV2\\
--bcg_reg_prob 0.2 --triplane_depth 3 --density_noise_fade_kimg 200 --density_reg 0 --min_yaw 0 --max_yaw 180 --back_repeat 4 --trans_reg 10 --gpc_reg_prob 0.7\\
--neural_rendering_resolution_final=128 --resume_kimg 1000
```




## Acknowledgements

We thank Metabrix lab for this highly realistic multi_pose generation.

