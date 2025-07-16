# Visual Tracking with Fully Convolutional Networks (FCNT)

## Overview

This project is a Python reimplementation based on the ICCV 2015 paper [Visual Tracking with Fully Convolutional Networks](https://scott89.github.io/FCNT/) by *Lijun Wang, Wanli Ouyang, Xiaogang Wang and Huchuan Lu*.

## Official Implementation

The Official MATLAB implementation can be accessed via the [paper's GitHub repository](https://github.com/scott89/FCNT).

---

## 🛠️ Configuration

Before running the tracker, you must edit the config file:

```
configs/config.yaml
```

In particular, change these values:

```yaml
sequence_path: data/sample_sequence/      # Path to your video sequence
init_bbox: [x, y, w, h]                   # Initial bounding box in the first frame
```

## 🔧 Environment Setup

This project uses **Conda**. Run the script below to create the environment and run the code after configuring `config.yaml` or you can use the command in section [Run Tracker](#-run-tracker):


```bash
bash run_fcnt.sh
```

## 🚀 Run Tracker

Once the config file is set, you can also run the tracker as follow:

```
python run.py --config configs/config.yaml
```