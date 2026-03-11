# 上海大学信号与系统2课程项目A

一个基于 **DeepFilterNet2** 的音频降噪工具，带可视化界面，新手友好教程。
除此之外，本程序还有基于mmse的传统降噪，程序中有对两种降噪效果的对比

## 功能

- 加载音频文件
- 进行 MMSE / DeepFilterNet2 降噪
- 显示降噪前后波形对比

---

## 安装与运行（适合新手）

### 1. 克隆仓库

```bash
$ git clone https://github.com/你的用户名/DeepFilterNet2-GUI.git
$ cd DeepFilterNet2-GUI
```

### 2. 创建虚拟环境（推荐 Python 3.10）

```bash
# Windows
$ python -m venv venv
$ venv\Scripts\activate

# macOS / Linux
$ python3 -m venv venv
$ source venv/bin/activate
```

终端提示激活成功后，会看到前面有 `(venv)`：

```bash
(venv) $
```

---

### 3. 安装依赖

```bash
(venv) $ pip install --upgrade pip
(venv) $ pip install -r requirements.txt
```

输出示例：

```bash
Collecting torch
...
Successfully installed torch torchaudio matplotlib numpy
```

---

### 4. 运行 GUI 程序

```bash
(venv) $ python run_gui.py
```

如果一切正常，会弹出图形界面，你可以：

- 上传音频文件
- 点击“开始降噪”按钮
- 查看降噪前后的波形对比
- 保存处理后的音频

---

### 5. 命令行降噪示例（非 GUI）

你也可以直接在终端跑：

```bash
(venv) $ python denoise_audio.py --input example.wav --output example_denoised.wav
```

输出示例：

```bash
Loading model...
Processing example.wav ...
Denoising completed: example_denoised.wav
```

---

### 6. 注意事项

1. 建议使用 **Python 3.10 **  
2. 确保虚拟环境已激活再运行程序  
3. 如果遇到 `No module named 'torchaudio.backend'`，请确认 PyTorch 与 Torchaudio 版本匹配  



### 7. 致谢

本项目基于 [DeepFilterNet2](https://github.com/EakAip/DeepFilterNet2#) 开发，感谢原作者提供优秀模型。
