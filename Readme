# TVGCN: Time-varying Graph Convolutional Networks for Multivariate and Multi-feature Spatio-temporal Series Prediction


This is a PyTorch implementation of TVGCN in the paper entitled "TVGCN: Time-varying Graph Convolutional Networks for Multivariate and Multi-feature Spatio-temporal Series Prediction"

Using a novel time-varying graph convolutional network (TVGCN) model based on attention at all stages (including training, validating and testing stage) for multi-features spatio-temporal series prediction, also, using absolute time embedding to improve the accuracy.


#### Datasets
The PEMS04 and PEMS08 datasets are collected by the Caltrans Performance Measurement System ([PeMS](http://pems.dot.ca.gov/)) ([Chen et al., 2001](https://trrjournalonline.trb.org/doi/10.3141/1748-12)) in real time every 30 seconds. The traffic data are aggregated into every 5-minute interval from the raw data. The system has more than 39,000 detectors deployed on the highway in the major metropolitan areas in California. Geographic information about the sensor stations are recorded in the datasets. There are three kinds of traffic measurements considered in our experiments, including total flow, average speed, and average occupancy.
1. PEMS-04: 

   traffic data collected from 307 detectors with 3 features, namely flow, occupy and speed. [1]

2. PEMS-08: 

   traffic data collected from 170 detectors with 3 features, namely flow, occupy and speed. [1]

3. toy:

    sine-wave sequences with multi-variety, which has strong dependence. [2]


Evaluation results on four real-world datasets shows that our model consistently outputs state-of-the-art results.

## Step 1 Create data directories
mkdir -p data/PEMS04
mkdir -p data/PEMS08
mkdir -p data/toy
## Step 2 Data Preparation
Copy each raw data to PEMS04, PEMS08 and toy directories
### PEMS04
python generate_training_data.py --output_dir=data/PEMS04 --filename=data/PEMS04/pems04.npz --seq_length_x=12 --seq_length_y=12
### PEMS08
python generate_training_data.py --output_dir=data/PEMS08 --filename=data/PEMS08/pems08.npz --seq_length_x=12 --seq_length_y=12
### Toy without time embedding
cd data/toy
python generate_npy.py
cd ../../
python generate_training_data.py --output_dir=data/toy --filename=data/toy/toy.npz --seq_length_x=12 --seq_length_y=12
### Toy with time embedding
cd data/toy
python genratedate_emb.py
cd ../../
python generate_training_data.py --output_dir=data/toy --filename=data/toy/toy_all.npz --seq_length_x=12 --seq_length_y=12
## Step 3 Train Commands
python train.py

#### References
[1] Guo, Shengnan,  et al. “Attention based spatial-temporal graph convolutional networks for traffic flow forecasting.” Proceedings of the AAAI conference on artificial intelligence. Vol. 33. No. 01. 2019.

[2] Shih, Shun-Yao, Fan-Keng Sun, and Hung-yi Lee. "Temporal pattern attention for multivariate time series forecasting." Machine Learning 108.8 (2019): 1421-1441.

[3] Li, Mengzhang, and Zhanxing Zhu. "Spatial-temporal fusion graph neural networks for traffic flow forecasting." Proceedings of the AAAI conference on artificial intelligence. Vol. 35. No. 5. 2021.

[4] Wu, Zonghan, et al. "Graph wavenet for deep spatial-temporal graph modeling." arXiv preprint arXiv:1906.00121 (2019).

[5] Song, Chao, et al. "Spatial-temporal synchronous graph convolutional networks: A new framework for spatial-temporal network data forecasting." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 34. No. 01. 2020.
