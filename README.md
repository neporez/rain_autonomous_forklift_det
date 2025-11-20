## Requirements

- **Jetpack**: 6.2
- **CUDA**: 12.6
- **PyTorch**: 2.8.0
- **ROS2**: Humble

## Installation

### 1. Jetson AGX Orin Setup

```bash
sudo apt update
sudo apt upgrade
sudo apt install python3-pip
sudo -H pip install -U jetson-stats
sudo nvpmodel -m0
```

**Note**: Reboot after setting power mode.

```bash
sudo jetson_clocks --store
```

### 2. Workspace Setup

#### Install Dependencies

```bash
sudo apt install git
sudo apt install ros-humble-tf-transformations
```

#### Clone Repository

```bash
mkdir -p ~/rain_ws/src
git clone https://github.com/neporez/rain_autonomous_forklift_det.git
mv ~/rain_autonomous_forklift_det/rain_autonomous_forklift_det ~/rain_ws/src
mv ~/rain_autonomous_forklift_det/RobDet3D ~
```

#### Build CUMM

```bash
cd ~
git clone https://github.com/FindDefinition/cumm.git
cd cumm
git checkout v0.7.11
sudo pip install -e .
```

#### Build SpConv

```bash
export CUMM_CUDA_ARCH_LIST="8.7"
cd ~
git clone https://github.com/traveller59/spconv.git
cd spconv
sudo pip install -e .
```

#### Install RobDet3D

```bash
cd ~/RobDet3D
pip install -r requirements.txt
python setup.py develop --user
```

#### Build Plugin

```bash
cd ~/RobDet3D/plugin
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
```

#### Prepare Dataset and Checkpoint

Download the required files:
- [Dataset and Checkpoint](https://drive.google.com/drive/folders/1i03jYwg8Uacrsz5wv8MMO7OA7s6m6Zzd?usp=sharing)

```bash
cd ~/RobDet3D
mkdir data
mv [your/dataset/path] ~/RobDet3D/data
mv [your/checkpoint/path] ~/RobDet3D
```

#### Export to TensorRT

```bash
cd ~/RobDet3D
python tools/deploy/export_onnx.py \
    --cfg_file configs/iassd/iassd_hvcsx2_gq_4x8_80e_warehouse.py \
    --ckpt checkpoint_epoch_80.pth \
    --onnx tools/models/trt

python tools/deploy/export_trt.py \
    --cfg_file configs/iassd/iassd_hvcsx2_gq_4x8_80e_warehouse.py \
    --onnx tools/models/trt/iassd_hvcsx2_gq_4x8_80e_warehouse.onnx \
    --batch 1 \
    --type FP16

mv ~/RobDet3D/tools/models/trt/iassd_hvcsx2_gq_4x8_80e_warehouse_FP16.engine \
    ~/rain_ws/src/rain_autonomous_forklift_det/rain_autonomous_forklift_det/model/checkpoint/warehouse_FP16.engine
```

### 3. ROS2 Package Build

```bash
cd ~/rain_ws
rosdep init
rosdep update
rosdep install --from-paths src -y --ignore-src
colcon build --symlink-install
```

## Usage

```bash
source ~/rain_ws/install/setup.bash
ros2 launch rain_autonomous_forklift_det rain_autonomous_forklift_det.launch.py
```
