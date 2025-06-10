# EvINR_towards_fastevent - 使用指南

## 快速开始

1. **克隆仓库**
   ```bash
   git clone https://github.com/111495zjy/EvINR_towards_fastevent.git
2. **解压高速数据集**
   ```bash
   unzip /content/drive/MyDrive/gun_bullet_mug.zip -d /content/
3.**安装依赖**
   ```bash
   cd /content/EvINR_towards_fastevent

4**将txt转为npy格式**
   ```bash
   pip install -r requirements.txt


   python /content/EvINR_towards_fastevent/txt_npy.py

   python train.py -n /content/EvINR_towards_fastevent -d /content/EvINR_towards_fastevent/gun_bullet_mug.npy


```
@article{wang2024EvINR,
  title={Revisit Event Generation Model: Self-Supervised Learning of Event-to-Video Reconstruction with Implicit Neural Representations},
  author={Wang, Zipeng and Lu, Yunfan and Wang, Lin},
  journal={ECCV},
  year={2024}
}
```
