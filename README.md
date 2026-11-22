# KITTI 目标分类的相机与激光雷达融合

给定 KITTI 二维目标框，识别汽车、行人和骑行者。每个目标由左彩色图像与投影到图像的激光雷达点描述。两路核分类输出类别分布，再按入框点数分配雷达权重。雷达缺失时使用相机分支。

## 环境

Python 3.12。以下命令在仓库根目录运行。

```sh
python -m venv .venv
```

Windows PowerShell 使用 `.venv\Scripts\python.exe -m pip install -r requirements.txt`。macOS 或 Linux 使用 `.venv/bin/python -m pip install -r requirements.txt`。激活虚拟环境后可直接执行下列 `python` 命令。

## 单帧演示

```sh
python download.py --demo
python features.py --demo
```

程序读取帧 000000 的图像、点云、标定与标签，输出目标类别、入框点数和两路描述子维数。完成训练后，运行 python demo.py 可查看同一帧的两路预测、融合预测和权重。

## 完整复现

```sh
python download.py
python features.py
python experiment.py
```

下载脚本按 `split.csv` 从 KITTI 的 AWS 开放归档读取图像与点云的 ZIP 成员，并解包对应标定与标签。照片和点云占用约 1.7 GB 磁盘空间。文件放入 `data/image_2`、`data/velodyne`、`data/calib` 和 `data/label_2`。删除自定义清单后再次运行可按固定的驾驶序列分组和种子重建列表。

`features.py` 输出 `results/features.npz`，包含每一帧的目标特征、类别、折号和入框点数。`experiment.py` 学习两路分类器和分档权重，更新 `results/metrics.json`、`results/predictions.csv` 和小样本推理所用的 `results/model.npz`。

- 折 0 用于测试，折 1–4 用于训练与折外权重估计。
- 完整观测下比较单相机、单雷达、等权融合和自适应融合。
- 缺失试验以测试帧为单位遮蔽雷达，比较直接混合与相机回退。分类指标为 Accuracy、逐类 F1 和 Macro-F1。

## 数据出处

- [KITTI 目标检测基准](https://www.cvlibs.net/datasets/kitti/eval_object.php)
- [AWS 开放数据归档](https://registry.opendata.aws/kitti/)
- A. Geiger, P. Lenz, and R. Urtasun, The KITTI Vision Benchmark Suite, CVPR 2012. [DOI](https://doi.org/10.1109/CVPR.2012.6248074)
- KITTI 数据及其衍生样本适用 [CC BY-NC-SA 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/)。
