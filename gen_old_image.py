import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import os

def add_scratches(image, num_scratches=10):
    """Thêm vết xước ngẫu nhiên vào hình ảnh."""
    if len(image.shape) == 3:  # Ảnh màu
        height, width, _ = image.shape
    else:  # Ảnh xám
        height, width = image.shape

    for _ in range(num_scratches):
        x1, y1 = random.randint(0, width-1), random.randint(0, height-1)
        x2, y2 = random.randint(0, width-1), random.randint(0, height-1)
        thickness = random.randint(1, 3)
        color = (random.randint(200, 255),) * (3 if len(image.shape) == 3 else 1)
        cv2.line(image, (x1, y1), (x2, y2), color, thickness)
    return image

def add_noise(image, intensity=20):
    """Thêm nhiễu hạt vào hình ảnh."""
    if len(image.shape) == 3:  # Ảnh màu
        noise = np.random.randint(-intensity, intensity, image.shape, dtype='int16')
        noisy_image = cv2.add(image.astype('int16'), noise, dtype=cv2.CV_8U)
    else:  # Ảnh xám
        noise = np.random.randint(-intensity, intensity, image.shape, dtype='int16')
        noisy_image = cv2.add(image.astype('int16'), noise, dtype=cv2.CV_8U)
    return noisy_image

def add_stains(image, num_stains=3):
    """Giả lập các vết ố trên hình ảnh."""
    if len(image.shape) == 3:  # Ảnh màu
        height, width, _ = image.shape
    else:  # Ảnh xám
        height, width = image.shape

    for _ in range(num_stains):
        x, y = random.randint(0, width-1), random.randint(0, height-1)
        radius = random.randint(20, 50)
        intensity = random.randint(50, 100)
        color = (intensity,) * (3 if len(image.shape) == 3 else 1)
        overlay = image.copy()
        cv2.circle(overlay, (x, y), radius, color, -1)
        alpha = random.uniform(0.3, 0.7)  # Độ mờ
        image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    return image

def vintage_effect(image_path):
    """Chuyển đổi hình ảnh sang hiệu ứng cũ."""
    # Đọc ảnh
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image is None:
        raise ValueError("Không thể tải hình ảnh. Kiểm tra đường dẫn.")

    # Thêm nhiễu
    noisy_image = add_noise(image, intensity=random.randint(1, 30))

    # Thêm vết xước
    scratched_image = add_scratches(noisy_image, num_scratches=random.randint(2, 10))
    
    # Không cần thêm vết ố và không chuyển đổi định dạng màu nữa
    # Chỉ cần trả về ảnh đã thêm xước
    return scratched_image


def ScratchMask(image, min_line_length=20, max_line_gap=5, mask_size=(256, 256)):
    """
    Tạo một mask với các vết xước dựa trên ảnh đầu vào.

    Args:
        image (ndarray): Ảnh đầu vào (grayscale hoặc RGB).
        min_line_length (int): Chiều dài tối thiểu của các đường thẳng.
        max_line_gap (int): Khoảng cách tối đa giữa các đoạn để nối thành một đường.
        mask_size (tuple): Kích thước của mask (mặc định là 256x256).

    Returns:
        mask (ndarray): Mặt nạ với các vết xước (0: vết xước, 1: vùng không bị che).
    """
    # Nếu ảnh là RGB, chuyển sang grayscale
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    # Áp dụng Thresholding để phân biệt vùng sáng và tối
    _, thresh_image = cv2.threshold(gray_image, 240, 255, cv2.THRESH_BINARY)

    # Tìm cạnh sử dụng Canny Edge Detector
    edges = cv2.Canny(thresh_image, 50, 150)

    # Tìm các đường thẳng sử dụng phép biến đổi Hough Line
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50,
                            minLineLength=min_line_length, maxLineGap=max_line_gap)

    # Khởi tạo mask trắng với kích thước `mask_size` (giá trị mặc định là 1.0)
    mask = np.ones(mask_size, dtype=np.float32)

    # Vẽ các đường thẳng (vết xước) lên mask
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Tính toán tỉ lệ giữa kích thước gốc và mask
            scale_x = image.shape[1] / mask_size[1]
            scale_y = image.shape[0] / mask_size[0]
            # Điều chỉnh tọa độ để vẽ chính xác lên mask
            x1, y1, x2, y2 = int(x1 / scale_x), int(y1 / scale_y), int(x2 / scale_x), int(y2 / scale_y)
            cv2.line(mask, (x1, y1), (x2, y2), color=0.0, thickness=2)  # Vẽ đường thẳng với giá trị 0.0 (vết xước)

    return mask


def main( input_image_path):
    vintage_image = vintage_effect(input_image_path)
    vintage_image = cv2.resize(vintage_image, (512, 512))
    base_name, extension = os.path.splitext(input_image_path)
    vintage_image_path = f'{base_name}_vintage_image{extension}'
    plt.imsave(vintage_image_path, vintage_image)

    mask = ScratchMask(vintage_image, min_line_length=30, max_line_gap=10, mask_size=(256, 256))
    mask_path = f'{base_name}_mask{extension}'
    cv2.imwrite(mask_path, (mask * 255).astype(np.uint8))  # Chuyển về uint8 để lưu

    return vintage_image_path, mask_path