## Install requirements
Experiments were conducted on Linux. Requirements may be different on Windows or MacOS

1. Create a new Python virtual environment using Python 3.9
1. Install pytorch 
```
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```
2. Install mmdetection
```
pip install -U openmim
mim install mmengine
mim install "mmcv-full==1.7.2"
pip install "mmsegmentation=0.30.0"
```
3. install other libraries using pip
```
python3 -m pip install \
   check_orientation==0.0.5 \
   cmake==3.29.3 \
   einops==0.8.0 \
   h5py==3.11.0 \
   kneed==0.8.5 \
   kornia==0.7.2 \
   kornia-moons==0.2.9 \
   kornia-rs==0.1.3 \
   opencv-python==4.9.0.80 \
   pandas==2.0.3 \
   pygraphviz==1.11 \
   scipy==1.10.1 \
   scikit-learn==1.3.2 \
   timm==1.0.3 \
   tqdm==4.66.4 \
   transformers==4.40.1 \
   yacs==0.1.8
```
4. install pycolmap
```
git clone https://github.com/colmap/colmap
cd colmap
mkdir build
cd build
cmake .. -GNinja -DCMAKE_CUDA_ARCHITECTURES=86
ninja
sudo ninja install
```
5. install lightglue
```
git clone https://github.com/cvg/LightGlue.git && cd LightGlue
python -m pip install -e .
```


## Preparation
1. Clone this repository
   ```
   git clone https://github.com/tmyok/kaggle-image-matching-challenge-2024.git
   ```
2. Download the datasets into the input directory. For more details, refer to [input/README.md](input/README.md).

## Inference
For the test data:
```
cd working
python3 inference.py
```

Evaluation:
```
cd /kaggle/working
python3 evaluate.py
```