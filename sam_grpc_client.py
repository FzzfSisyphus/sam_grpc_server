import grpc
import sam_service_pb2
import sam_service_pb2_grpc
from PIL import Image
import matplotlib.pyplot as plt
import io
import os
import numpy as np 
import cv2 


def process_image(image_path,output_dir='output_masks',server_address='localhost:50051'):
    """发送图片到 gRPC 服务器并获取结果"""
    # 读取图片
    with open(image_path, 'rb') as f:
        image_data = f.read()
    
    MAX_MESSAGE_LENGTH = 2000 * 1024 * 1024
    options = [
        ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
        ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
    ]

    # 创建 gRPC 通道
    channel = grpc.insecure_channel(server_address, options=options)
    stub = sam_service_pb2_grpc.SAMServiceStub(channel)
    
    # 创建请求
    request = sam_service_pb2.ImageRequest(image_data=image_data)
    
    # 发送请求
    try:
        print("Sending request to server...")
        response = stub.ProcessImage(request)
        
        print(f"\nReceived {response.total_masks} masks")
        print(f"Image dimensions: {response.image_width} x {response.image_height}")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存每个 mask
        for idx, mask in enumerate(response.masks):
            # 从字节数据恢复 numpy 数组
            # 重要：需要知道原始形状来正确恢复数组
            mask_array = np.frombuffer(mask.mask_array, dtype=bool)
            mask_array = mask_array.reshape(response.image_height, response.image_width)
            
            # 转换为 0-255 的灰度图
            mask_uint8 = (mask_array * 255).astype(np.uint8)
            
            # 转换为 PIL Image 并保存为 PNG
            mask_image = Image.fromarray(mask_uint8, mode='L')
            mask_path = os.path.join(output_dir, f'mask_{idx:04d}.png')
            mask_image.save(mask_path)
            
            print(f"Mask {idx}: area={mask.area:.0f}, "
                  f"bbox=({mask.bbox.x}, {mask.bbox.y}, {mask.bbox.width}, {mask.bbox.height}), "
                  f"iou={mask.predicted_iou:.3f}, "
                  f"stability={mask.stability_score:.3f}")
            
            if (idx + 1) % 10 == 0:
                print(f"Saved {idx + 1}/{response.total_masks} masks")
        
        print(f"\nAll masks saved to {output_dir}/")
        
        return response
        
    except grpc.RpcError as e:
        print(f"RPC failed: {e.code()}: {e.details()}")
        return None
    finally:
        channel.close()

def visualize_masks(image_path, masks_dir='output_masks', output_path='visualization.png', 
                   borders=True, show_original=True, dpi=150):
    """可视化所有 masks（基于 show_anns 的风格）"""
    
    # 加载所有 masks
    mask_files = sorted([f for f in os.listdir(masks_dir) if f.endswith('.png')])
    
    if len(mask_files) == 0:
        print("No masks found!")
        return
    
    print(f"Loading {len(mask_files)} masks...")
    
    # 先加载第一个 mask 来获取正确的尺寸
    first_mask_path = os.path.join(masks_dir, mask_files[0])
    first_mask = np.array(Image.open(first_mask_path))
    mask_height, mask_width = first_mask.shape
    
    print(f"Mask dimensions: {mask_width} x {mask_height}")
    
    # 加载原始图片并调整到 mask 的尺寸
    original_image = np.array(Image.open(image_path))
    print(f"Original image dimensions: {original_image.shape[1]} x {original_image.shape[0]}")
    
    # 如果尺寸不匹配，调整原始图片
    if original_image.shape[0] != mask_height or original_image.shape[1] != mask_width:
        print(f"Resizing original image from {original_image.shape[1]}x{original_image.shape[0]} to {mask_width}x{mask_height}")
        original_image = np.array(Image.fromarray(original_image).resize((mask_width, mask_height)))
    
    # 加载 masks 并构建 anns 结构
    anns = []
    for idx, mask_file in enumerate(mask_files):
        mask_path = os.path.join(masks_dir, mask_file)
        mask = np.array(Image.open(mask_path)) > 0
        area = np.sum(mask)
        anns.append({
            'segmentation': mask,
            'area': area
        })
        
        if (idx + 1) % 10 == 0:
            print(f"Loaded {idx + 1}/{len(mask_files)} masks")
    
    # 按面积排序（从大到小）
    sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
    print(f"Sorted masks by area (largest: {sorted_anns[0]['area']:.0f} pixels)")
    
    # 创建可视化
    if show_original:
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        
        # 显示原图
        axes[0].imshow(original_image)
        axes[0].set_title('Original Image', fontsize=16, fontweight='bold')
        axes[0].axis('off')
        
        # 在第二个子图上绘制
        ax = axes[1]
    else:
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # 显示原图作为背景
    ax.imshow(original_image)
    ax.set_autoscale_on(False)
    
    # 创建透明层
    img = np.ones((mask_height, mask_width, 4))
    img[:, :, 3] = 0
    
    # 设置随机种子以保持颜色一致性
    np.random.seed(3)
    
    print("Drawing masks...")
    
    # 叠加所有 masks
    for idx, ann in enumerate(sorted_anns):
        m = ann["segmentation"]
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask
        
        if borders:
            # 找到轮廓
            contours, _ = cv2.findContours(
                m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            # 平滑轮廓
            contours = [
                cv2.approxPolyDP(contour, epsilon=0.01, closed=True)
                for contour in contours
            ]
            # 绘制轮廓（蓝色边界）
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1)
        
        if (idx + 1) % 10 == 0:
            print(f"Drew {idx + 1}/{len(sorted_anns)} masks")
    
    ax.imshow(img)
    ax.set_title(f'All Masks Overlay ({len(mask_files)} masks)', 
                 fontsize=16, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    
    print(f"Saving visualization to {output_path}...")
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"Visualization saved successfully!")
    plt.close()

def get_mask_info(masks_dir='output_masks'):
    """获取所有 mask 的信息"""
    mask_files = sorted([f for f in os.listdir(masks_dir) if f.endswith('.png')])
    
    print(f"\n{'='*60}")
    print(f"Total masks: {len(mask_files)}")
    print(f"{'='*60}")
    
    for idx, mask_file in enumerate(mask_files):
        mask_path = os.path.join(masks_dir, mask_file)
        mask = np.array(Image.open(mask_path))
        
        # 计算统计信息
        white_pixels = np.sum(mask > 0)
        total_pixels = mask.size
        percentage = (white_pixels / total_pixels) * 100
        
        print(f"Mask {idx:04d}: {white_pixels:8d} pixels ({percentage:5.2f}%)")

if __name__ == '__main__':
    # 测试
    # result = process_image('sam3test.png')
    # image='groceries.jpg'
    # image='sam3test.png'
    image='rokea.jpg'

    result = process_image(image)
    if result:
        # 显示 mask 信息
        get_mask_info()
        
        # 可视化结果
        visualize_masks(image)
