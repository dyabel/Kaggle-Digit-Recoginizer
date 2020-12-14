# README

## 环境需求

- python==3.7

- torch==1.7.1

  其余见requirement.txt

## 使用方法

先pip install -r requirement.txt

- cnn

  划分验证集

  python main.py --net 0 --all_data 0 --config_path cnn_config.json或者./run_cnn.sh

  用全部数据训练

  python main.py --net 0 --all_data 1 --config_path cnn_config.json或者修改./run_cnn.sh中的all_data

- mlp

  划分验证集

  python main.py --net 1 --all_data 0 --config_path mlp_config.json或者./run_mlp.sh

  用全部数据训练

  python main.py --net 1 --all_data 1 --config_path mlp_config.json或者修改./run_mlp.sh中的all_data

- lstm

  划分验证集

  python main.py --net 2 --config_path lstm_config.json或者./run_lstm.sh

  用全部数据训练

  python main.py --net 2 --all_data 1 --config_path lstm_config.json或者修改./run_lstm.sh中的all_data

  

  结果保存在pred.json中

  