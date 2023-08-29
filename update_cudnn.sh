export URL="https://developer.nvidia.com/compute/cudnn/secure/8.6.0/local_installers/11.8/cudnn-linux-x86_64-8.6.0.163_cuda11-archive.tar.xz"
# ==== DOWNLOAD CUDDN ==== 
curl $URL -o ./cudnn-linux-x86_64-8.6.0.163_cuda11-archive.tar.xz 
sudo tar -xvf ./cudnn-linux-x86_64-8.6.0.163_cuda11-archive.tar.xz
# ==== INSTALL CUDDN ==== 
sudo cp ./cudnn-*-archive/include/cudnn*.h /usr/local/cuda/include 
sudo cp -P ./cudnn-*-archive/lib/libcudnn* /usr/local/cuda/lib64 
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
# ==== CONFIGURE DYNAMIC RUNTIME BINDINGS ==== 
sudo ldconfig
# ==== INSTALL CONDA ENV ==== 
conda create -n "tfgpu" python=3.10 -y
conda activate tfgpu
conda install -c conda-forge cudatoolkit=11.8.0 ipykernel -y
python3 -m pip install nvidia-cudnn-cu11==8.6.0.163 tensorflow==2.12.*
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
python3 -m ipykernel install --user --name tfgpu --display-name "Python (tf-cudnn8.6)"
# ==== VERIFY ==== 
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

