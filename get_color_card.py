import cv2
import numpy as np
import os

image = None


# 鼠标回调函数
def mouse_callback(event, x, y, flags, param):
    global points
    image = param['image']
    filename = param['filename']
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow('Select Corners', image)
        if len(points) == 4:
            print("Four points selected. Processing image...")
            process_image(image, np.array(points, dtype=np.float32), filename)


# 透视变换函数
# 将src_image图像中src_points四个点的区域变换到一个400*600的矩形区域
def warp_perspective(src_image, src_points):
    # 按照左上，右上，左下，右下的排序src_points
    print(src_points)
    src_points = src_points[np.argsort(src_points.sum(axis=1))]
    if src_points[0][1] > src_points[1][1]:
        src_points[0], src_points[1] = src_points[1], src_points[0]
    if src_points[2][1] > src_points[3][1]:
        src_points[2], src_points[3] = src_points[3], src_points[2]
    print('Sorted points:', src_points)
    # 目标图像的四个点
    dst_points = np.array([[0, 0], [0, 400], [600, 0], [600, 400]],
                          dtype=np.float32)
    # 透视变换矩阵
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    # 透视变换
    warped_image = cv2.warpPerspective(src_image, M, (600, 400))
    # 显示透视变换后的图像
    cv2.imshow('Warped Image', warped_image)
    return warped_image


# 白平衡校正函数
def correct_white_balance(warped_image):
    # 这里应该添加白平衡校正逻辑
    return warped_image


# 图像处理函数
def process_image(image, points, filename):
    corrected_image = warp_perspective(image, points)
    # 提取corrected_image中的色块中心点颜色（4*6色块）
    nrows, ncols = 4, 6
    row_height = corrected_image.shape[0] // nrows
    col_width = corrected_image.shape[1] // ncols
    colors = []
    # 在每个色块中心点提取颜色
    # 绘制色块中心点
    for i in range(nrows):
        for j in range(ncols):
            x = j * col_width + col_width // 2
            y = i * row_height + row_height // 2
            color = corrected_image[y, x]
            colors.append(color)
            # cv2.circle(corrected_image, (x, y), 5, (0, 255, 0), -1)
    print(colors)
    # 显示处理后的图像
    cv2.imshow('Corrected Image', corrected_image)
    # 保存colors到csv文件
    output_filename = filename.replace('.jpg', '.csv')
    output_file_path = os.path.join('color_card', output_filename)
    if not os.path.exists('color_card'):
        os.makedirs('color_card')
    # 保存colors到csv文件
    np.savetxt(output_file_path, colors, delimiter=',', fmt='%d')


# 批处理图像函数
def batch_process_images(folder_path):
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            if image is not None:
                process_image_with_selection(image, filename)


# 选择点并处理图像函数
def process_image_with_selection(image, filename):
    global points
    points = []
    cv2.namedWindow('Select Corners')
    # 在调用cv2.setMouseCallback之前，将需要的数据封装在一个字典中
    data = {'image': image, 'filename': filename}
    # 修改cv2.setMouseCallback的调用方式，传递封装的数据
    cv2.setMouseCallback('Select Corners', mouse_callback, data)
    cv2.imshow('Select Corners', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def average_color_card(folder):
    colors = []
    for filename in os.listdir(folder):
        if filename.lower().endswith('.csv'):
            color = np.loadtxt(os.path.join(folder, filename), delimiter=',')
            colors.append(color)
    colors = np.array(colors)
    average_color = np.mean(colors, axis=0)
    print(average_color)
    return average_color


if __name__ == "__main__":
    folder_path = 'data'
    batch_process_images(folder_path)

    avg_color = average_color_card('color_card')
    df = pd.DataFrame(avg_color[:, ::-1])
    df.columns = ['R', 'G', 'B']
    df.to_csv('average_color_card.csv', index=False, sep='\t')
