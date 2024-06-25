# 照片白平衡归一化

一个小功能，用于将照片的白平衡调整到一致。

## 使用方法

1. 将照片放入 `./data` 文件夹中

2. 运行 `get_color_card.py` 文件，生成白平衡参考卡的颜色数据，输出到 `./color_card` 中对应文件名的 `.csv` 文件中

    ```bash
    python get_color_card.py
    ```

3. 运行 `main.py` 文件进行最终的白平衡调整。最终的照片将输出到 `./correct` 文件夹中

    ```bash
    python main.py
    ```
