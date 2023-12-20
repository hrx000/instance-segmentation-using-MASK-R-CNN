# Head Hair removal: Removing Hair from Portraits Using GANs

Head Hair removal is a hair-removal network that can be applied in hair design and 3D face reconstruction.


<img width="767" alt="hair_removal" src="https://github.com/hrx000/instance-segmentation-using-MASK-R-CNN/assets/51284717/7373f77c-ef5e-46bc-abf9-1d623559546d">

 

**Abstract:**

Removing hair from portrait images is challenging due to the complex occlusions between hair and face, as well as the lack of paired portrait data with/without hair. To this end, we present a dataset and a baseline method for removing hair from portrait images using generative adversarial networks (GANs). Our core idea is to train a fully connected network ** Head Hair Removal** to find the direction of hair removal in the latent space of StyleGAN for the training stage. 

## Requirements

1. Windows (not tested on Linux yet)
2. Python 3.7
3. NVIDIA GPU + CUDA11.1 + CuDNN

# Install

1. ```bash
   git clone git@github.com:oneThousand1000/HairMapper.git
   ```
   
1. Download the following pretrained models, put each of them to **path**:

   | model                                                        | path                               |
   | ------------------------------------------------------------ | ---------------------------------- |
   | [StyleGAN2-ada-Generator.pth](https://drive.google.com/file/d/1EsGehuEdY4z4t21o2LgW2dSsyN3rxYLJ/view?usp=sharing) | ./ckpts                            |
   | [e4e_ffhq_encode.pt](https://drive.google.com/file/d/1cUv_reLE6k3604or78EranS7XzuVMWeO/view) | ./ckpts                            |
   | [model_ir_se50.pth](https://drive.google.com/file/d/1GIMopzrt2GE_4PG-_YxmVqTQEiaqu5L6/view?usp=sharing) | ./ckpts                            |
   | [face_parsing.pth](https://drive.google.com/file/d/1IMsrkXA9NuCEy1ij8c8o6wCrAxkmjNPZ/view?usp=sharing) | ./ckpts                            |
   | [vgg16.pth](https://drive.google.com/file/d/1EPhkEP_1O7ZVk66aBeKoFqf3xiM4BHH8/view?usp=sharing) | ./ckpts                            |
   | [classification_model.pth](https://drive.google.com/file/d/1SSw6vd-25OGnLAE0kuA-_VHabxlsdLXL/view?usp=sharing) | ./classifier/gender_classification |
   | [classification_model.pth](https://drive.google.com/file/d/1n14ckDcgiy7eu-e9XZhqQYb5025PjSpV/view?usp=sharing) | ./classifier/hair_classification   |

​	face_parsing.pth from: https://github.com/switchablenorms/CelebAMask-HQ/tree/master/face_parsing ([79999_iter.pth](https://drive.google.com/file/d/1eP90uPItdAy1czivugAM3ZK68OdY2pfe/view?usp=sharing))

​	e4e_ffhq_encode.pt from: https://github.com/omertov/encoder4editing

​	model_ir_se50.pth from: https://github.com/orpatashnik/StyleCLIP

The StyleGAN2-ada-Generator.pth contains the same model parameters as the original [stylegan2](https://github.com/NVlabs/stylegan2) pkl model `stylegan2-ffhq-config-f.pkl`.

2. Create conda environment:

   ```
   conda create -n HairMapper python=3.7
   activate HairMapper
   ```

3. [**StyleGAN2-ada requirements**](https://github.com/NVlabs/stylegan2-ada-pytorch): The code relies heavily on custom PyTorch extensions that are compiled on the fly using NVCC. On Windows, the compilation requires Microsoft Visual Studio. We recommend installing [Visual Studio Community Edition](https://visualstudio.microsoft.com/vs/) and adding it into `PATH` using `"C:\Program Files (x86)\Microsoft Visual Studio\<VERSION>\Community\VC\Auxiliary\Build\vcvars64.bat"`.

  
3. Then install other dependencies by

   ```
   pip install torch===1.7.1+cu110 torchvision===0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
   ```

   ```
   pip install -r requirements.txt
   ```

# Models

Please fill out this google form for pre-trained models access:

https://forms.gle/a5pRbE3yxEr7sZDm7

Then download and put the pre-trained models to **path**:

| model                                                | path                                   |
| ---------------------------------------------------- | -------------------------------------- |
| Final Head Hair removal (can be applied to female and male) | mapper/checkpoints/final/best_model.pt |
| Man Head Hair removal (can only be applied to male)         | mapper/checkpoints/man/best_model.pt   |



# Testing

Directly use our pre-trained model for hair removal.

**step1:**

Real images **should be extracted and aligned using DLib and a function from the original FFHQ dataset preparation step**, you can use the [image align code](https://github.com/Puzer/stylegan-encoder/blob/master/align_images.py) provided by [stylegan-encoder](https://github.com/Puzer/stylegan-encoder).

Please put the aligned real images to **./test_data/origin** (examplar data can be found in ./data/test_data/final/origin).

**step2:**

Then using encoder4editing to get the corresponding latent codes:

```
cd encoder4editing
python encode.py  --data_dir ../test_data
```

latent codes will be saved to `./test_data/code`.

**step3:**

Then run HairMapper:

```python
cd ../
python main_mapper.py  --data_dir ./test_data
```

If you want to perform an additional diffusion (slower, but can achieve better results):

```python
python main_mapper.py  --data_dir ./test_data --diffuse
```


## Reference and Acknowledgements

We thank the MetaBrix Lab for their good work.




