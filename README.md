# Saidi Anomaly Inference API

This is an Inference API repo for saidi anomaly detection.

## Usage

1. Install Python 3.10 with necessary requirements. If you are using `Anaconda`, here is an example:

```shell
conda create -n alumina python=3.10 jupyter notebook
pip install -r requirements.txt
```

2. Prepare Checkpoint. We offer a checkpoint trained by TimesNet in route `checkpoints`. Feel free to add your own checkpoint in this route.
3. Prepare Data. Place the dataset is in route `./data/anomaly.csv`.
4. Run inference. The forecasting script is briefed in file `inference_example.py`

## Contact

If you have any questions or suggestions, feel free to contact us:

- Yunzhong Qiu (Master student, qiuyz24@mails.tsinghua.edu.cn)