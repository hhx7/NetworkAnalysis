##环境安装
conda env create -f=./src/requirements.txt -n network 

##切换环境
conda activate network

##ML算法
python ./src/NetworkAnalysis_ML.py

##DL算法
python ./src/NetworkAnalysis_DL.py
