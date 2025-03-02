pip install functorch==2.0.0
pip install scipy
pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-2.0.1+cu117.html
pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-2.0.1+cu117.html
pip install --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/torch-2.0.1+cu117.html
pip install --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-2.0.1+cu117.html
pip install torchrl==0.0.5
pip install torch-geometric==2.0.1
pip install wandb==0.13.9
pip install pytorch-lightning==2.0.1
pip install texttable==1.7.0
pip install lkh==1.1.1
pip install tsplib95==0.7.1
pip install einops==0.7.0
pip install gym==0.26.2
pip install hydra-core==1.3.2
pip install lightning==2.0.0
pip install pygmtools
wget http://akira.ruc.dk/~keld/research/LKH-3/LKH-3.0.7.tgz
tar xvfz LKH-3.0.7.tgz
cd LKH-3.0.7
make
cd ..
cp -r LKH-3.0.7 ~/miniconda3/envs/ml4tsp/bin
