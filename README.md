# EvINR_towards_fastevent - 使用指南

## 快速开始

1. **克隆仓库**
   ```
   git clone https://github.com/111495zjy/EvINR_towards_fastevent.git
   ```
2. **解压高速数据集**
   ```
   #根据电脑地址修改
   unzip /content/drive/MyDrive/gun_bullet_mug.zip -d /content/
   ```
3.**将txt文件转为npy文件**
   ```
   #根据电脑地址修改
   python /content/EvINR_towards_fastevent/txt_npy.py
   ```
4.**安装依赖**
   ```
   #根据电脑地址修改
   cd /content/EvINR_towards_fastevent
   pip install -r requirements.txt
   pip install lpips
   ```
   
6.**训练模型**
   ```
   #根据电脑地址修改
python train.py -n /content/EvINR_towards_fastevent -d /content/EvINR_towards_fastevent/gun_bullet_mug.npy
   ```

notes(代码中需要修改的文件地址):
   ```
            #输出图片的地址
            train.py line 85:output_path = os.path.join('/content/EvINR_towards_fastevent/logs', 'output_image_{}.png'.format(i))
   ```
```
@article{wang2024EvINR,
  title={Revisit Event Generation Model: Self-Supervised Learning of Event-to-Video Reconstruction with Implicit Neural Representations},
  author={Wang, Zipeng and Lu, Yunfan and Wang, Lin},
  journal={ECCV},
  year={2024}
}
```
