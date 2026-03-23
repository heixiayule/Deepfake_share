## Deepfake Audio Detection


<p align="center">
    <img src="diagram.png" alt="method flow chart" width="800">
</p>

[![](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=darkgreen)](https://www.python.org) [![](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=TensorFlow&logoColor=white)](https://www.tensorflow.org) [![](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=Keras&logoColor=white)](https://keras.io)  [![](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/stable/) [![](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org) [![](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org) ![Matplotlib](https://img.shields.io/badge/Matplotlib-%25dd99.svg?style=for-the-badge&logo=Matplotlib&logoColor=black) [![](https://img.shields.io/badge/librosa-FFD499?style=for-the-badge)](https://librosa.org/doc/latest/index.html#) 

The misuse of the latest powerful generative algorithms poses a threat to individuals and society, as public opinion can be swayed through the spread of modified content, especially deepfaked audio. In this project, we developed a detection system to distinguish between bonafide and spoofed audio.


### Setup
Install the required dependencies. Download the GPU version of these package if required.
``` bash
pip install -r requirements.txt
```
Download and unzip AsvSpoof 2019 LA files

```bash
wget https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip
unzip LA.zip
```

This should give you the following dir:
``` bash
audio-deepfake
├── LA
│   ├── ASVspoof2019_LA_asv_protocols
│   ├── ASVspoof2019_LA_asv_scores
│   ├── ASVspoof2019_LA_cm_protocols
│   ├── ASVspoof2019_LA_dev
│   │   └── flac
│   ├── ASVspoof2019_LA_eval
│   │   └── flac
│   └── ASVspoof2019_LA_train
│       └── flac
└── src
    ├── feature.py
    ├── metrics.py
    ├── augment.py
    ├── model
    ├── protocol
    ├── requirements.txt
    └── run.py

```

Before running the experiment, remember to change the path to include your uniqname:

```
path_to_database = "/home/[uniqname]/audio-deepfake-detection/" + access_type
```

Train the model and evaluate on AsvSpoof 2019 eval dataset by the following command. EER, Accuracy, F1, precision, recall and auc score are provided.

``` bash
python run.py \
    -m 'lcnn' \
    -f 'cqt' \
    --lr 0.00001 \
    --epochs 100 \
    --batch 32 \
```

Before you run the experiement on the full dataset, you can set the dataset size to 1000 and verbose to 1 for quick verification.
```bash
python run.py \
    -m 'lcnn' \
    -f 'cqt' \
    --lr 0.00001 \
    --epochs 100 \
    --batch 32 \
    --datasize 1000 \
    --verbose 1 \
    --savedata False \
```

Aditionally, you can use the augment.py code to perform data augmentation. When training the model, just add the `--augment 1` argument.

### 数据流流程图

想快速了解从数据准备到结果导出的完整流程？请查阅：

👉 **[docs/flowchart.md](docs/flowchart.md)**

该文档使用 **Mermaid** 绘制了训练/推理的端到端数据流，涵盖：
- 数据准备与协议 CSV 读取
- STFT/CQT 特征提取及 `__cache__/*.npz` 缓存机制
- 可选增强数据集（`--augment 1`）的拼接方式
- 模型构建（`lcnn` / `lcnn-lstm`）与训练回调
- 评估推理、EER 计算及 `__log__/predictions-*.csv` 导出

> **如何阅读：** 在 GitHub 上直接打开 [docs/flowchart.md](docs/flowchart.md)，Mermaid 流程图会自动渲染；也可在支持 Mermaid 的编辑器（如 VS Code + Mermaid 插件）中本地预览。

