import os
import pandas as pd
import numpy as np
import cv2

COLOR_CARD_STANDARD = pd.read_csv('color_card.csv', delimiter='\t')
print(COLOR_CARD_STANDARD)
RGB = COLOR_CARD_STANDARD[['R', 'G', 'B']].values


# 绘制色卡
def draw_color_card(rgb=RGB):
    color_card = np.zeros((400, 600, 3), dtype=np.uint8)
    for i in range(4):
        for j in range(6):
            x = j * 100
            y = i * 100
            color_rgb = rgb[i * 6 + j]
            color_bgr = color_rgb[::-1]
            color_card[y:y + 100, x:x + 100] = color_bgr
    cv2.imshow('Color Card', color_card)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# draw_color_card()

def correct_white_balance(image, card_bgr):
    # 将图片转换到BGR颜色空间，因为OpenCV默认使用BGR
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # 假设您已经有了色卡的原始RGB值和图片中对应的RGB值
    # 这里用随机数生成一些示例数据
    original_rgb_values = RGB
    observed_rgb_values = card_bgr[:, ::-1]

    # 计算转换矩阵
    # 首先将RGB值标准化到[0, 1]区间
    original_rgb_values_norm = original_rgb_values / 255.0
    observed_rgb_values_norm = observed_rgb_values / 255.0

    # 使用线性回归计算转换矩阵
    # 这里使用最小二乘法
    A = np.hstack((np.ones((24, 1)), observed_rgb_values_norm))
    x, _, _, _ = np.linalg.lstsq(A, original_rgb_values_norm, rcond=None)

    # 应用转换矩阵到整张图片
    image_norm = image_bgr.astype(np.float32) / 255.0
    # 创建一个与image_norm在前两个维度上匹配，但在第三维上为1的偏置项数组
    bias = np.ones((image_norm.shape[0], image_norm.shape[1], 1))

    # 使用np.concatenate而不是np.hstack，并在颜色通道维度上堆叠
    image_transformed = np.dot(np.concatenate((bias, image_norm), axis=2), x)

    # 其余代码保持不变
    image_transformed = np.clip(image_transformed * 255, 0, 255).astype(np.uint8)

    # 将处理后的图像转换回BGR颜色空间
    image_corrected = cv2.cvtColor(image_transformed, cv2.COLOR_RGB2BGR)

    # 保存或显示结果
    cv2.imshow('White Balanced Image', image_corrected)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return image_corrected

for file in os.listdir('color_card'):
    if file.endswith('.csv'):
        observed_bgr_values = pd.read_csv(os.path.join('color_card', file),
                                          header=None).values
        image = cv2.imread(os.path.join('data', file.replace('.csv', '.jpg')))
        correct_img = correct_white_balance(image, observed_bgr_values)
        # save the corrected image
        cv2.imwrite(os.path.join('correct', file.replace('.csv', '_corrected.jpg')), correct_img)
