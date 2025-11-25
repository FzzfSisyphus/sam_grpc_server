import os
import io
import grpc
from concurrent import futures
import numpy as np
import torch
from PIL import Image
import sam_service_pb2
import sam_service_pb2_grpc

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


class SAMServicer(sam_service_pb2_grpc.SAMServiceServicer):
    def __init__(self):
        """初始化 SAM 模型"""
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        
        sam2_checkpoint = "../checkpoints/sam2.1_hiera_base_plus.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
        
        sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
        sam2 = torch.compile(sam2, mode="max-autotune")
        
        self.mask_generator = SAM2AutomaticMaskGenerator(
            model=sam2,
            points_per_side=32,
            points_per_batch=8,
            pred_iou_thresh=0.7,
            stability_score_thresh=0.92,
            stability_score_offset=0.7,
            crop_n_layers=1,
            box_nms_thresh=0.7,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=25.0,
            use_m2m=False,
        )
        print("SAM model initialized successfully")
    
    def ProcessImage(self, request, context):
        """处理图片请求"""
        try:
            # 从二进制数据加载图片
            image_bytes = io.BytesIO(request.image_data)
            image = Image.open(image_bytes)
            image = np.array(image.convert("RGB"))
            
            height, width = image.shape[:2]
            print(f"Processing image of shape: {image.shape}")
            
            # 生成 masks
            
            torch.cuda.synchronize()
            
            masks = self.mask_generator.generate(image)
            
            torch.cuda.synchronize()
            
            
            print(f"Generated {len(masks)} masks")
            
            # 构建响应
            response_masks = []
            
            for idx, mask_data in enumerate(masks):
                # 直接将 numpy 数组转换为字节
                # mask_data['segmentation'] 是 boolean numpy 数组
                mask_array = mask_data['segmentation']
                
                # 使用 numpy 的压缩二进制格式
                mask_bytes = mask_array.tobytes()
                
                # 获取边界框
                bbox = mask_data['bbox']  # [x, y, w, h]
                
                # 创建 Mask 消息
                mask_msg = sam_service_pb2.Mask(
                    mask_array=mask_bytes,
                    area=float(mask_data['area']),
                    bbox=sam_service_pb2.BoundingBox(
                        x=int(bbox[0]),
                        y=int(bbox[1]),
                        width=int(bbox[2]),
                        height=int(bbox[3])
                    ),
                    predicted_iou=float(mask_data['predicted_iou']),
                    stability_score=float(mask_data['stability_score'])
                )
                
                response_masks.append(mask_msg)
                
                if (idx + 1) % 10 == 0:
                    print(f"Processed {idx + 1}/{len(masks)} masks")
            
            # 创建响应
            response = sam_service_pb2.MaskResponse(
                masks=response_masks,
                image_height=height,
                image_width=width,
                total_masks=len(masks)
            )
            
            # 计算响应大小
            response_size = response.ByteSize()
            print(f"Response size: {response_size / (1024 * 1024):.2f} MB")
            
            return response
            
        except Exception as e:
            import traceback
            print(f"Error processing image: {str(e)}")
            print(traceback.format_exc())
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f'Error: {str(e)}')
            return sam_service_pb2.MaskResponse()

def serve():
    """启动 gRPC 服务器"""
    MAX_MESSAGE_LENGTH = 2000 * 1024 * 1024 
    
    options = [
        ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
        ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
    ]

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10),
                         options=options)
    
    sam_service_pb2_grpc.add_SAMServiceServicer_to_server(
        SAMServicer(), server
    )
    
    server_address = '[::]:50051'
    server.add_insecure_port(server_address)
    server.start()
    
    print(f"SAM gRPC Server started on {server_address}")
    print(f"Max message size: {MAX_MESSAGE_LENGTH / (1024 * 1024):.0f}MB")

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.stop(0)


if __name__ == '__main__':
    serve()