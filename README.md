# SSU
modular saccades for scene understanding architecture



Please note that installing the GPU version of TensorFlow in the Docker image does not guarantee that it will work with your host system's GPU. You need to ensure that your host system has the required NVIDIA drivers and CUDA Toolkit installed. Additionally, you need to have the `nvidia-docker2` package installed on your host system and use the `--gpus all` flag when running the container (`docker run --gpus all -it <gpu-image-name>`).

To install `nvidia-docker2` on your host system, follow these steps:
1. Remove older versions of NVIDIA Docker, if any:
```bash
sudo apt-get remove -y nvidia-docker
```
2. Add the official NVIDIA package repositories for Docker:
```bash
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
```
3. Update the package index:
```bash
sudo apt-get update
```
4. Install `nvidia-docker2` and reload the Docker daemon configuration:
```bash
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```