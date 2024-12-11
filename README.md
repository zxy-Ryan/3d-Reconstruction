## **Preparation**

Experiments were conducted on Linux. Requirements may vary for Windows or macOS.

### **1. Clone the Repository**
```
git clone https://github.com/zxy-Ryan/3d-Reconstruction.git
```

### **2. Download Datasets**
Place the datasets in the \`input\` directory. Refer to [input/README.md](input/README.md) for detailed instructions.

### **3. Create a Python Virtual Environment**
Ensure you are using Python 3.9.

### **4. Install PyTorch**
To ensure compatibility with your CUDA version, please refer to the [PyTorch installation guide](https://pytorch.org/get-started/locally/). Below is an example installation command for **CUDA 11.8**:
```
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```

### **5. Install MMSegmentation**
```
pip install -U openmim
mim install mmengine
mim install "mmcv-full==1.7.2"
pip install "mmsegmentation==0.30.0"
```

### **6. Install Additional Libraries**
```
pip install \
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
   scipy==1.10.1 \
   scikit-learn==1.3.2 \
   timm==1.0.3 \
   tqdm \
   transformers==4.40.1 \
   yacs==0.1.8 \
   pycolmap==0.6.1
```

### **7. Install LightGlue**
```
git clone https://github.com/cvg/LightGlue.git && cd LightGlue
python -m pip install -e .
```

### **8. Install 3D Reconstruction Visualization Dependencies**
**Note:** Some Linux systems may not support installing `rerun-sdk`. This has been tested successfully on macOS 14.1 without any issues.
```
pip install rerun-sdk requests
```

---

## **Usage**

### **1. Navigate to the Working Directory**
```
cd working
```

### **2. Inference**
- If necessary, update the \`Config\` class in \`inference.py\`. 
- Run:
  ```
  python3 inference.py
  ```

### **3. Evaluation**
- Update \`input_dir\`, \`output_dir\`, and \`gt_dir\` (ground truth) in \`evaluate.py\` if needed.  
- Run:
  ```
  python3 evaluate.py
  ```

### **4. Visualization**

#### **Feature Matching**
- Update \`matches_path\`, \`image_dir\`, \`output_dir\`, and \`keypoints_path\` in \`matching_visualization.py\` if needed.  
- Run:
  ```
  python3 matching_visualization.py
  ```

#### **3D Reconstruction**
- Update \`path_to_church\` and \`path_to_church_images\` in \`3D_real.py\` if necessary.  
- Run:
  ```
  python3 3D_real.py
  ```

---

## **Demo**

To run a complete demo, follow the steps above with default configurations. Run the following commands sequentially, and results will appear in the \`output\` folder:

```
cd working
python3 inference.py
python3 evaluate.py
python3 matching_visualization.py
python3 3D_real.py
```

### **Runtime Information**
- **ALIKED method:** ~2 hours on a single V100 GPU.
- **SIFT method:** ~45 minutes on a single V100 GPU.  
The default setting uses the **ALIKED method**.

---

### **Notes**
- Ensure your device meets the CUDA architecture requirements (set to 86 in this example).
- Configurations can be updated in respective \`.py\` files to customize inputs and outputs.

---

