# openvla-7b-inference-implement
python setup and script for openvla-7b inference

## 🎯Step1: create virtual environment & download openvla github repo
details at: https://github.com/openvla/openvla
```
# Create and activate conda environment
conda create -n openvla python=3.10 -y
conda activate openvla

# Install PyTorch. Below is a sample command to do this, but you should check the following link
# to find installation instructions that are specific to your compute platform:
# https://pytorch.org/get-started/locally/
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y  # UPDATE ME!

# Clone and install the openvla repo
git clone https://github.com/openvla/openvla.git
cd openvla
pip install -e .
```

## 🎯Step2: download pretrained openvla-7b model
details at: https://huggingface.co/openvla/openvla-7b
```
powershell -ExecutionPolicy ByPass -c "irm https://hf.co/cli/install.ps1 | iex"
hf download openvla/openvla-7b
```

## 🎯Step3: download and run the inference script
```
gh repo clone Ccyyrrrrr/openvla-7b-inference-implement
cd yourpath/to/inference/script
python inference.py
```
😁strongly recommand u to storage the inference.py under the openvla folder
