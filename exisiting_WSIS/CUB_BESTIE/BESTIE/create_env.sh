pip install uv
uv venv --python 3.09 --clear .run_venv
source .run_venv/bin/activate
uv pip install "numpy<2.0.0"
ln -s $(python -c "import numpy; print(numpy.get_include())")/numpy .run_venv/include/numpy
uv pip install "setuptools<70" wheel
uv pip install chainercv==0.13.1 --no-build-isolation
uv pip install -r requirements.txt
uv pip install pycocotools
uv pip install "numpy<1.20.0"
uv pip install  --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install  --no-cache-dir numpy scipy matplotlib Pillow tqdm tensorboard
uv pip install opencv-python-headless
uv pip install scikit-learn
uv pip install "numpy<1.23.0"
mkdir -p .run_venv/include
uv pip install tqdm
uv pip install yacs
