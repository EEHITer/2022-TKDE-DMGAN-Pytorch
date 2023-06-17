## DMGAN: Dynamic Multi-Hop Graph AttentionNetwork for Traffic Forecasting
![avatar](/images/DMGAN_architecture.png)
This is a PyTorch implementation of Diffusion Convolutional Recurrent Neural Network in the following paper:
Rui Li, Fan Zhang, Tong Li, Ning zhang, Tingting Zhang, https://drive.google.com/file/d/1VMDLjTeYjSwsY35IISFAR3njDV-ImfYB/view, TKDE 2022.

# Citation
@inproceedings{li2018dcrnn_traffic,
  title={Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting},
  author={Li, Yaguang and Yu, Rose and Shahabi, Cyrus and Liu, Yan},
  booktitle={International Conference on Learning Representations (ICLR '18)},
  year={2018}
}


# Data Preparation
The traffic data files for Los Angeles (METR-LA) and the Bay Area (PEMS-BAY), i.e., `metr-la.h5` and `pems-bay.h5`, are available at [Google Drive](https://drive.google.com/open?id=10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX) or [Baidu Yun](https://pan.baidu.com/s/14Yy9isAIZYdU__OYEQGa_g), and should be put into the `data/` folder.
The `*.h5` files store the data in `panads.DataFrame` using the `HDF5` file format. Here is an example:

|                     | sensor_0 | sensor_1 | sensor_2 | sensor_n |
|:-------------------:|:--------:|:--------:|:--------:|:--------:|
| 2018/01/01 00:00:00 |   60.0   |   65.0   |   70.0   |    ...   |
| 2018/01/01 00:05:00 |   61.0   |   64.0   |   65.0   |    ...   |
| 2018/01/01 00:10:00 |   63.0   |   65.0   |   60.0   |    ...   |
|         ...         |    ...   |    ...   |    ...   |    ...   |


Here is an article about [Using HDF5 with Python](https://medium.com/@jerilkuriakose/using-hdf5-with-python-6c5242d08773).

Run the following commands to generate train/test/val dataset at  `data/{METR-LA,PEMS-BAY}/{train,val,test}.npz`.
```bash
# Create data directories
mkdir -p data/{METR-LA,PEMS-BAY}

# METR-LA
python -m scripts.generate_training_data --output_dir=data/METR-LA --traffic_df_filename=data/metr-la.h5

# PEMS-BAY
python -m scripts.generate_training_data --output_dir=data/PEMS-BAY --traffic_df_filename=data/pems-bay.h5
```

# Model
--dmgagru_cell.py: The implementation of DMGA-GRUcell. 
--dmgan_model.py: The encoder-decoder framework of DMGAN.
--Transformer.py: The implementation of global-attention or local attention.
--dmgan_supervisor.py: The arichteriture of the Network for training and testing.



## Model Training
```bash
# METR-LA
python train.py --config_filename=data/DMGAN_la.yaml

# PEMS-BAY
python train.py --config_filename=data/DMGAN_bay.yaml
```
