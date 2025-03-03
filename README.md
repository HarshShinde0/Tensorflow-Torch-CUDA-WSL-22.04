# Install TensorFlow GPU on WSL2 Ubuntu 24.04 (Windows 10/11) | CUDA, cuDNN, TensorRT & PyTorch -- GPU

## Prerequisites
Ensure you have the latest version of WSL2 and Ubuntu 24.04 installed. Also, make sure you have Windows 10/11 with the latest updates.

## Step 1: System Update
```bash
sudo apt update
sudo apt upgrade -y
sudo apt install build-essential -y
```

## Step 2: Install Miniconda
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

## Step 3: Install CUDA
```bash
wget https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda_12.1.1_530.30.02_linux.run
sudo sh cuda_12.1.1_530.30.02_linux.run
```

Add CUDA to your PATH and LD_LIBRARY_PATH:
```bash
nano ~/.bashrc
```
Add the following lines to `.bashrc`:
```bash
export PATH=/usr/local/cuda-12.1/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```
Source the `.bashrc` file:
```bash
source ~/.bashrc
```

Update `ld.so.conf` and configure dynamic linker run-time bindings:
```bash
sudo nano /etc/ld.so.conf
```
Add the following line to `/etc/ld.so.conf`:
```bash
/usr/local/cuda-12.1/lib64
```
Then run:
```bash
sudo ldconfig
```

Verify the installation:
```bash
echo $PATH
echo $LD_LIBRARY_PATH
sudo ldconfig -p | grep cuda
nvcc --version
```

## Step 4: Install cuDNN
Download cuDNN from [NVIDIA cuDNN Archive](https://developer.nvidia.com/rdp/cudnn-archive).

Extract and copy cuDNN files:
```bash
tar -xvf cudnn-linux-x86_64-8.9.7.29_cuda12-archive.tar.xz
cd cudnn-linux-x86_64-8.9.7.29_cuda12-archive
sudo cp include/cudnn*.h /usr/local/cuda-12.1/include
sudo cp lib/libcudnn* /usr/local/cuda-12.1/lib64
sudo chmod a+r /usr/local/cuda-12.1/include/cudnn*.h /usr/local/cuda-12.1/lib64/libcudnn*
cd ..
```

Optional: Test cuDNN installation:
```bash
nano test_cudnn.c
```
Add the following code to `test_cudnn.c`:
```c
#include <cudnn.h>
#include <stdio.h>

int main() {
    cudnnHandle_t handle;
    cudnnStatus_t status = cudnnCreate(&handle);
    if (status == CUDNN_STATUS_SUCCESS) {
        printf("cuDNN successfully initialized.\n");
    } else {
        printf("cuDNN initialization failed.\n");
    }
    cudnnDestroy(handle);
    return 0;
}
```
Compile and run the test:
```bash
gcc -o test_cudnn test_cudnn.c -I/usr/local/cuda-12.1/include -L/usr/local/cuda-12.1/lib64 -lcudnn
./test_cudnn
```

## Step 5: Install TensorRT
Download TensorRT from [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt/download).

Extract and copy TensorRT files:
```bash
tar -xzvf TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-12.0.tar.gz
sudo mv TensorRT-8.6.1.6 /usr/local/TensorRT-8.6.1
```

Add TensorRT to your PATH and LD_LIBRARY_PATH:
```bash
nano ~/.bashrc
```
Add the following lines to `.bashrc`:
```bash
export PATH=/usr/local/cuda-12.1/bin:/usr/local/TensorRT-8.6.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:/usr/local/TensorRT-8.6.1/lib:$LD_LIBRARY_PATH
```
Source the `.bashrc` file:
```bash
source ~/.bashrc
```

## Step 6: Create Conda Environment and Install TensorFlow
```bash
conda create --name tf python=3.9 -y
conda activate tf
```

Install TensorFlow with GPU support:
```bash
python -m pip install tensorflow[and-cuda]
```

Verify TensorFlow installation:
```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## Step 7: Install TensorRT Python Bindings
```bash
cd /usr/local/TensorRT-8.6.1/python
pip install tensorrt-8.6.1-cp39-none-linux_x86_64.whl
pip install tensorrt_dispatch-8.6.1-cp39-none-linux_x86_64.whl
pip install tensorrt_lean-8.6.1-cp39-none-linux_x86_64.whl
```

Verify TensorRT installation:
```python
import tensorflow as tf
print(tf.version)
```

## Step 8: Install JupyterLab
```bash
pip install jupyterlab
jupyter lab
```

## Step 9: Install PyTorch with GPU Support
```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121
```

Verify PyTorch installation:
```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

By following these steps, you will have TensorFlow, TensorRT, and PyTorch set up with GPU support on WSL2 Ubuntu 24.04.
