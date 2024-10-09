# run below cmd first
# conda activate AGNOMIN_Decoder

# conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
# conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio cudatoolkit=11.8 -c pytorch -c nvidia -y
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia


pip install transformers
pip install scipy
pip install pandas
pip install cython
pip install wandb
pip install scikit-learn
pip install sentence-transformers
pip install seaborn
pip install numpy==1.21.0
pip install numba==0.56

git clone https://github.com/kunaldahiya/pyxclib
cd pyxclib
pip install .
cd ..

## Need only for apex optimizers
git clone https://github.com/NVIDIA/apex 
cd apex
git checkout 5c9625cfed681d4c96a0ca4406ea6b1b08c78164
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cuda_ext" ./
cd ..

