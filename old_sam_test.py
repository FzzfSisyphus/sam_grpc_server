import os

# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
from sam2codebase.sam2.build_sam import build_sam2
from sam2codebase.sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import cv2
from realsense import RealSense
from robot_arm_toolbox.xarm7_hai_vacuum_gripper import Xram7_Gripper
import time
import open3d as o3d
from loguru import logger
from datetime import datetime

ready_pose_camera_to_base = np.array(
    [
        [0.04648286, -0.99885982, 0.01088311, 0.30828422],
        [-0.99891591, -0.04645245, 0.00303029, 0.04245176],
        [-0.00252129, -0.01101217, -0.99993616, 0.40913597],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


class sam2_auto_mask:
    def __init__(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")

            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

        sam2_checkpoint = "./sam2codebase/checkpoints/sam2.1_hiera_large.pt"
        # model_cfg = "sam2codebase/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
        # model_cfg = "/sam2codebase/configs/sam2.1/sam2.1_hiera_l.yaml"
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

        sam2 = build_sam2(
            model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False
        )
        # self.mask_generator = SAM2AutomaticMaskGenerator(sam2)
        # self.mask_generator = SAM2AutomaticMaskGenerator(
        #     model=sam2,
        #     points_per_side=64,
        #     points_per_batch=128,
        #     pred_iou_thresh=0.7,
        #     stability_score_thresh=0.92,
        #     stability_score_offset=0.7,
        #     crop_n_layers=1,
        #     box_nms_thresh=0.7,
        #     crop_n_points_downscale_factor=2,
        #     min_mask_region_area=25.0,
        #     use_m2m=True,
        # )
        self.mask_generator = SAM2AutomaticMaskGenerator(
            model=sam2,
            points_per_side=16,  # 原来是64，减少到32，大幅减少点数
            points_per_batch=32,  # 降低 batch，减少 GPU 占用峰值
            pred_iou_thresh=0.5,  # 降低预测质量阈值，略提速
            stability_score_thresh=0.85,  # 降低稳定性阈值，减少 mask 数
            stability_score_offset=0.6,  # 可略调小
            crop_n_layers=0,  # 不使用图像裁剪层，提升速度非常明显
            box_nms_thresh=0.8,  # 提高 NMS 阈值，减少重复 masks，提速
            crop_n_points_downscale_factor=2,  # 保持默认即可
            min_mask_region_area=100.0,  # 提高 mask 面积下限，跳过小目标
            use_m2m=False,  # 关闭 Multi-mask-to-Mask 推理，加速
        )
        self.camera = RealSense(serial="337122076130", frame_rate=30)

        self.camera_intrinsics = {
            "fx": self.camera.color_intrin_part[0],
            "fy": self.camera.color_intrin_part[1],
            "cx": self.camera.color_intrin_part[2],
            "cy": self.camera.color_intrin_part[3],
        }

        # self.check_camera = RealSense(serial="337122073741", frame_rate=30)
        # self.check_camera.init_stereo("logs/stereo_model/model_best_bp2.pth")
        self.camera.init_stereo("logs/stereo_model/model_best_bp2.pth")

        self.robot = Xram7_Gripper(
            host="192.168.1.204", gripper_type="xarm_vacuum", global_cam=False
        )
        self.boxMinX = 0.0
        self.boxMaxX = 0.2
        self.boxMinY = 0.00
        self.boxMaxY = 0.05
        self.boxMinZ = 0.15
        self.boxMaxZ = 0.45
        self.DEBUG = False
        # self.DEBUG = True

        folder_name = datetime.now().strftime("%m-%d-%H-%M-%S")
        self.log_path = f"suction_sam_logs/{folder_name}"
        self.grasp_count = 0
        self.pre_grasp_success = True
        self.pre_grasp_position = None
        os.makedirs(self.log_path, exist_ok=True)

        log_file_path = os.path.join(self.log_path, "record.log")
        self.logger = logger
        self.logger.remove()
        self.logger.add(
            log_file_path,
            format=(
                "<green>{time:MM-DD HH:mm:ss}</green> | "
                "<level>{level: <8}</level>|"
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan>|"
                "<yellow>Pick#{extra[grasp_count]}</yellow> - "
                "<level>{message}</level>"
            ),
            colorize=True,
        )

        self.logger = self.logger.bind(grasp_count=self.grasp_count)

    def show_anns(self, anns, borders=True):
        np.random.seed(3)
        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
        ax = plt.gca()
        ax.set_autoscale_on(False)

        img = np.ones(
            (
                sorted_anns[0]["segmentation"].shape[0],
                sorted_anns[0]["segmentation"].shape[1],
                4,
            )
        )
        img[:, :, 3] = 0
        for ann in sorted_anns:
            m = ann["segmentation"]
            color_mask = np.concatenate([np.random.random(3), [0.5]])
            img[m] = color_mask
            if borders:
                contours, _ = cv2.findContours(
                    m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
                )
                # Try to smooth contours
                contours = [
                    cv2.approxPolyDP(contour, epsilon=0.01, closed=True)
                    for contour in contours
                ]
                cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1)

        ax.imshow(img)

    def calculate_priority_score(
        self,
        mask_info,
        max_area,
        min_area,
        max_mean_depth,
        min_mean_depth,
        area_weight=0.55,
        depth_weight=0.45,
    ):
        score = mask_info["area"] / mask_info["mean_depth"]
        if mask_info["mean_depth"] > 660:
            score = 0.0
        # print(f"area:{mask_info['area']}  center_depth:{mask_info['center_depth']} ,normal_area_score:{normalized_area_score} -- normalized_depth_score:{normalized_depth_score}")
        return score

    def get_mask(self, image):
        masks = self.mask_generator.generate(image)
        return masks

    def keep_largest_component(self,mask):
        """保留最大的连通组件，其余设为False"""
        # 转换为uint8格式进行连通组件分析
        binary_mask = mask.astype(np.uint8) * 255
        
        # 连通组件分析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_mask, connectivity=8
        )
        
        if num_labels <= 1:  # 只有背景，没有前景
            return np.zeros_like(mask, dtype=bool)
        
        # 找到最大的组件（跳过背景索引0）
        areas = stats[1:, 4]  # 获取所有组件的面积
        largest_component_idx = np.argmax(areas) + 1  # +1因为跳过了背景
        
        # 创建只包含最大组件的mask
        largest_mask = (labels == largest_component_idx)
        
        return largest_mask

    def filter_overlapping_masks_efficient(self, valid_masks):
        """
        高效版本：使用位运算进行mask重叠过滤
        """
        filtered_masks = []

        for i, current_mask in enumerate(valid_masks):
            current_seg = current_mask["mask"].copy()

            # 与所有之前的mask进行重叠检测和移除
            for prev_mask in filtered_masks:
                # 移除与之前mask的重叠部分
                current_seg = current_seg & (~prev_mask["mask"])

            kernel = np.ones((9, 9), np.uint8)  # 5x5的腐蚀核，可以调整大小
            current_seg = cv2.erode(current_seg.astype(np.uint8), kernel, iterations=5)
            current_seg = current_seg.astype(bool)  # 转回布尔类型
            current_seg = self.keep_largest_component(current_seg)

            # 检查是否还有剩余区域
            total_pixels = current_seg.size  # 总像素数
            true_pixels = np.sum(current_seg)  # True像素数
            if true_pixels / total_pixels >= 0.005:  # 5%以上     
            # if np.any(current_seg):
                # 创建新的mask字典
                new_mask = current_mask.copy()
                new_mask["mask"] = current_seg
                new_mask["area"] = np.sum(current_seg)
                filtered_masks.append(new_mask)

                # print(f"保留 Mask {i}, 过滤后面积: {new_mask['area']}")
            else:
                pass
                # print(f"跳过 Mask {i}, 完全重叠")

        idx = 0

        if self.DEBUG:
            for mask_info in filtered_masks:
                final_mask = mask_info["mask"]
                area = mask_info["area"]

                # --- 保存当前 mask 的黑白图像 ---
                binary_mask = (final_mask.astype(np.uint8)) * 255  # 转成0和255
                log_save_path = os.path.join(self.log_path, str(self.grasp_count))
                os.makedirs(log_save_path, exist_ok=True)
                mask_filename = os.path.join(
                    log_save_path, f"fliter_mask_{idx + 1}_area_{area}.png"
                )
                cv2.imwrite(mask_filename, binary_mask)
                idx += 1

        return filtered_masks

    def depth_mapping(self, valid_masks, depth_image):
        # 对每个mask计算平均深度
        for mask_info in valid_masks:
            mask = mask_info["mask"]
            # 提取mask区域对应的深度值
            mask_depths = depth_image[mask]
            valid_depths = mask_depths[mask_depths > 0]
            if len(valid_depths) > 0:
                mean_depth = np.mean(valid_depths)
            else:
                mean_depth = 0.0  # 如果没有有效深度值

            # 添加到mask信息中
            mask_info["mean_depth"] = mean_depth

            # print(f"Mask area: {mask_info['area']}, Mean depth: {mean_depth:.2f}")

        return valid_masks

    def get_valid_mask(self, color_image, depth_image, step):
        # 1. 读取原图和深度图

        image_np = np.array(color_image)
        depth_map = np.array(depth_image)
        # 确保深度图尺寸与原图一致
        depth_map = cv2.resize(depth_map, (image_np.shape[1], image_np.shape[0]))

        # 2. 使用 SAM 处理原图
        masks = self.get_mask(image_np)

        # 3. 定义工作空间像素范围（例如图像中心 60% 区域）
        H, W = image_np.shape[:2]
        x_min, x_max = int(W * 0.24), int(W * 0.85)
        y_min, y_max = int(H * 0.21), int(H * 0.9)

        # 构造工作空间 mask（像素位置筛选）
        workspace_mask = np.zeros((H, W), dtype=bool)
        workspace_mask[y_min:y_max, x_min:x_max] = True

        # 4. 筛选 mask
        result_mask_image = image_np.copy()
        max_area = 0
        # max_center = None

        cmap = plt.get_cmap("tab20")
        num_colors = 20
        idx = 0

        # 设置面积阈值
        area_threshold = 16900  # 根据需要调整这个值

        # 收集所有有效的mask数据
        valid_masks = []

        for mask_data in masks:
            mask = mask_data["segmentation"]  # bool mask shape (H, W)
            # 筛选深度 < 0.7m (即 700mm)
            depth_valid = depth_map < 680
            # 筛选工作空间范围
            final_mask = mask & depth_valid & workspace_mask
            # 计算面积并筛选
            area = final_mask.sum()
            if area < area_threshold:  # 去掉小于阈值的mask
                continue

            if area == 0:
                continue

            # 计算中心点
            ys, xs = np.where(final_mask)
            center = (int(xs.mean()), int(ys.mean()))
            center_depth = depth_image[center[1], center[0]]

            # 保存有效mask的信息
            valid_masks.append(
                {
                    "mask": final_mask,
                    "area": area,
                    "center": center,
                    "center_depth": center_depth,
                }
            )


        # reverse=False ： 小 - 大
        valid_masks.sort(
            key=lambda x: x["area"], reverse=False
        )  # reverse=false 从小到大
        valid_masks = self.filter_overlapping_masks_efficient(valid_masks)
        valid_masks = self.depth_mapping(valid_masks, depth_image)

        max_area = max(m["area"] for m in valid_masks)
        min_area = min(m["area"] for m in valid_masks)
        max_mean_depth = max(m["mean_depth"] for m in valid_masks)
        min_mean_depth = min(m["mean_depth"] for m in valid_masks)

        valid_masks.sort(
            key=lambda x: self.calculate_priority_score(
                x, max_area, min_area, max_mean_depth, min_mean_depth
            ),
            reverse=True,
        )
        if self.DEBUG:
            # 绘制mask和中心点
            idx = 0
            for mask_info in valid_masks:
                final_mask = mask_info["mask"]
                center = mask_info["center"]
                area = mask_info["area"]

                # print(
                #     f"idx:{idx} : 中心点 = {mask_info['center']},面积 = {mask_info['area']} , mean_depth:{mask_info['mean_depth']}"
                # )

                # 为每个mask分配颜色
                color = np.array(cmap(idx % num_colors)[:3]) * 255
                color = color.astype(np.uint8)
                result_mask_image[final_mask] = color

                # --- 保存当前 mask 的黑白图像 ---
                binary_mask = (final_mask.astype(np.uint8)) * 255  # 转成0和255

                log_save_path = os.path.join(self.log_path, f"{self.grasp_count}")
                os.makedirs(log_save_path, exist_ok=True)
                mask_filename = os.path.join(
                    log_save_path, f"mask_{idx + 1}_area_{area}.png"
                )
                cv2.imwrite(mask_filename, binary_mask)

                if idx == 0:
                    # --- 保存当前 mask 的彩色图像 ---
                    colored_mask = np.zeros_like(result_mask_image)  # 创建空白图像
                    colored_mask[final_mask] = color  # 只在mask区域填充颜色
                    cv2.circle(colored_mask, center, 6, (0, 0, 255), -1)
                    colored_mask_filename = os.path.join(
                        log_save_path, f"mask_{idx}_area_{area}_colored.png"
                    )
                    cv2.imwrite(colored_mask_filename, colored_mask)

                # 在mask中心绘制红点
                cv2.circle(result_mask_image, center, 6, (0, 0, 255), -1)

                # 可选：在红点旁边添加序号文本
                cv2.putText(
                    result_mask_image,
                    str(idx + 1),
                    (center[0] + 10, center[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )
                mask_info["idx"] = idx

                idx += 1

        # 输出信息
        # for i, mask_info in enumerate(valid_masks):
        #     print(
        #         f"Mask {i + 1}: 面积 = {mask_info['area']}, 中心点 = {mask_info['center']}"
        #     )
        if self.DEBUG:
            # 显示或保存结果
            # cv2.imwrite("result_mask.png", result_mask_image)
            log_save_path = os.path.join(self.log_path, f"{self.grasp_count}")
            os.makedirs(log_save_path, exist_ok=True)
            mask_filename = os.path.join(
                log_save_path, f"{self.grasp_count}_final_choice.png"
            )
            cv2.imwrite(mask_filename, result_mask_image)
            cv2.imshow("Filtered Masks", result_mask_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return valid_masks

    def pixel_to_world_coordinate(self, pixel_x, pixel_y, depth_value):
        """
        将像素坐标转换为世界坐标

        Args:
            pixel_x: 像素x坐标
            pixel_y: 像素y坐标
            depth_value: 深度值 (mm)

        Returns:
            world_coordinates: (x, y, z) 世界坐标 (mm)
        """
        # 相机内参 (需要根据你的相机校准结果调整)
        # 这里需要你的相机内参矩阵，示例值：
        fx = self.camera.color_intrin.fx  # x方向焦距
        fy = self.camera.color_intrin.fy  # y方向焦距
        cx = self.camera.color_intrin.ppx  # 主点x坐标
        cy = self.camera.color_intrin.ppy  # 主点y坐标

        # 像素坐标转相机坐标
        camera_x = (pixel_x - cx) * depth_value / fx
        camera_y = (pixel_y - cy) * depth_value / fy
        camera_z = depth_value
        # 相机坐标转世界坐标 (需要相机外参矩阵)
        # 这里需要你的手眼标定结果，示例变换矩阵：
        # 你需要替换为实际的外参矩阵
        camera_to_world_matrix = np.array(
            [
                [0.05212967, -0.99852163, 0.01539683, 295.48535],
                [-0.9986375, -0.05216001, -0.00157504, 47.50127],
                [0.00237581, -0.01529374, -0.99988019, 408.23159],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        # 相机坐标系下的齐次坐标
        camera_coords = np.array([camera_x, camera_y, camera_z, 1.0])

        # 转换到世界坐标系
        world_coords = camera_to_world_matrix @ camera_coords

        return world_coords[:3]  # 返回 (x, y, z)

    def real_to_pixel(self, grasp_size_m, depth, fx, fy):
        w, h = grasp_size_m
        w_px = w * fx / depth
        h_px = h * fy / depth
        return (w_px, h_px)

    def is_grasp_inside_mask(self, mask, center, angle_deg, gripper_size_px):
        (w, h) = gripper_size_px
        rect = ((center[0], center[1]), (w, h), angle_deg)
        box = cv2.boxPoints(rect)
        box = np.int32(box)

        # 判断矩形是否完全落在 mask 中
        for x, y in box:
            if x < 0 or x >= mask.shape[1] or y < 0 or y >= mask.shape[0]:
                return False
            if mask[y, x] == 0:
                return False
        return True

    def visualize_grasps(
        self,
        mask,
        center_depth,
        rgb_image,
        center,
        grasp_size_m,
        camera_intrinsics,
        angle_step=10,
    ):
        fx, fy = camera_intrinsics["fx"], camera_intrinsics["fy"]
        cx, cy = center
        depth = center_depth
        angle_res = 0
        # breakpoint()
        if depth <= 0:
            print("无效深度")
            return rgb_image

        gripper_size_px = self.real_to_pixel(grasp_size_m, depth, fx, fy)

        img_vis = rgb_image.copy()

        for angle in range(-90, 90, angle_step):
            rect = ((cx, cy), gripper_size_px, angle)
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            color = (
                (0, 255, 0)
                if self.is_grasp_inside_mask(mask, (cx, cy), angle, gripper_size_px)
                else (0, 0, 255)
            )
            cv2.drawContours(img_vis, [box], 0, color, 2)
            if color == (0, 255, 0):
                angle_res = angle
                break

        # 标注抓取中心点
        cv2.circle(img_vis, (cx, cy), 5, (255, 0, 0), -1)
        return img_vis, angle_res

    def grasp(self, suction_position):

        start_grasp_time = time.time()
        self.robot.arm.set_position(
            yaw=int(suction_position[5]), relative=True, speed=300, wait=True
        )
        code = self.robot.arm.set_position(*suction_position, vel=200)
        print("grasp time:",time.time() - start_grasp_time) 
        self.way_target_time = time.time()
        print("arrive target time:",self.way_target_time - start_grasp_time) # 0.05
        # breakpoint()
        if code != 0:
            raise RuntimeError(
                f"before griper close 1: arm movement failed with code: {code}"
            )
        code = self.robot.arm.set_position(z=-10, relative=True, speed=50, wait=True)
        self.time_push = time.time()
        print("check action time:",self.time_push - self.way_target_time)  # 1.93
        if code != 0:
            raise RuntimeError(
                f"before griper close 2: arm movement failed with code: {code}"
            )

        code = self.robot.close_gripper(1, wait=True)
        time.sleep(0.2)
        code = self.robot.arm.set_position(z=60, relative=True, speed=50, wait=True)
        print("check action time:",time.time() - self.time_push) #2.08
        if code != 0:
            raise RuntimeError(
                f"before griper close 1: arm movement failed with code: {code}"
            )

        self.robot.close_gripper(2, wait=True)

        self.robot.open_gripper(0)
        print(f"grasp time: {time.time() - start_grasp_time}") # 4.57

    def check_gripper_has_object(self, points):
        """
        通过点云密度判断夹爪是否抓住物体
        """
        if len(points) == 0:
            return False, "没有检测到点云数据"

        # 计算点云密度
        point_count = len(points)
        box_volume = (
            (self.boxMaxX - self.boxMinX)
            * (self.boxMaxY - self.boxMinY)
            * (self.boxMaxZ - self.boxMinZ)
        )
        density = point_count / box_volume

        # 设置密度阈值（需要根据实际情况调整）
        density_threshold = 2500000.0  # 每立方米的点数

        has_object = density > density_threshold
        return has_object, f"点云密度: {density:.2f}, 阈值: {density_threshold}"

    def check_object_by_height_distribution(self, points):
        """
        通过Z轴高度分布判断是否有物体
        """
        if len(points) < 10:  # 点太少
            return False, "点云数据不足"

        z_values = points[:, 2]

        # 计算高度统计信息
        z_mean = np.mean(z_values)
        z_std = np.std(z_values)
        z_max = np.max(z_values)
        z_min = np.min(z_values)

        # 物体存在的判断条件
        height_range = z_max - z_min
        height_threshold = 0.01  # 1cm，根据实际物体大小调整

        has_object = height_range > height_threshold

        return has_object, {
            "height_range": height_range,
            "z_mean": z_mean,
            "z_std": z_std,
            "point_count": len(points),
        }

    def detect_gripper_object(self):
        """
        综合检测夹爪是否抓住物体
        """
        colors, points = self.check_camera.get_stereo_rgbd()
        points_z = points[:, 2]
        points_x = points[:, 0]
        points_y = points[:, 1]

        # 应用你的区域mask
        box_area_mask = (
            (points_x > self.boxMinX)
            & (points_x < self.boxMaxX)
            & (points_y > self.boxMinY)
            & (points_y < self.boxMaxY)
            & (points_z < self.boxMaxZ)
            & (points_z > self.boxMinZ)
        )

        filtered_points = points[box_area_mask].astype(np.float32)

        if len(filtered_points) == 0:
            return False, "检测区域内没有点云数据"

        # 方法1：密度检测
        has_object_density, density_info = self.check_gripper_has_object(
            filtered_points
        )

        # 方法2：高度分布检测
        # has_object_height, height_info = self.check_object_by_height_distribution(filtered_points)

        # 综合判断
        # has_object = has_object_density or has_object_height
        has_object = has_object_density

        if self.DEBUG:
            print(f"密度检测: {has_object_density} - {density_info}")
            # print(f"高度检测: {has_object_height} - {height_info}")
            print(f"最终结果: {'检测到物体' if has_object else '未检测到物体'}")

            # 可视化
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(filtered_points)
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
            o3d.visualization.draw_geometries([cloud, frame])

        return has_object, {
            "density_result": has_object_density,
            # 'height_result': has_object_height,
            "point_count": len(filtered_points),
        }

    def pre_grasp_check(self, suction_position):
        xyz_distance = np.linalg.norm(
            np.array(self.pre_grasp_position[:3]) - np.array(suction_position[:3])
        )
        if xyz_distance < 50 and self.pre_grasp_success is False:
            breakpoint()
            return True
        return False

    def pre_grasp_change(self, suction_position):
        try:
            code = self.robot.arm.set_position(*suction_position, vel=200, wait=True)
            # breakpoint()
            if code != 0:
                raise RuntimeError(
                    f"before griper close 1: arm movement failed with code: {code}"
                )
            # code = self.robot.arm.set_position(z=45,relative=True, speed =50, wait=True)
            if code != 0:
                raise RuntimeError(
                    f"before griper close 1: arm movement failed with code: {code}"
                )
            self.robot.close_gripper(2, wait=True)
            self.robot.waypoint(vel=300)
        except Exception as e:
            print(f"grasp fail 抓取过程中报错 自动重启 {e}")
            self.robot.arm.clean_error()
            self.robot.arm.motion_enable(True)
            self.robot.arm.set_state(state=0)
            self.robot.open_gripper(0)
            self.robot.waypoint()

    def max_depth_in_radius(self, depth_map, center_pixel, radius=10):
        """
        获取像素中心点指定半径圆形区域内的最大深度值

        参数:
        - depth_map: 深度图像
        - center_pixel: 中心点坐标 (x, y)
        - radius: 半径（像素）

        返回:
        - max_depth: 圆形区域内最大深度值
        """
        if depth_map is None:
            return 0

        h, w = depth_map.shape[:2]
        cx, cy = int(center_pixel[0]), int(center_pixel[1])

        # 检查中心点是否在图像范围内
        if cx < 0 or cx >= w or cy < 0 or cy >= h:
            return 0

        max_depth = 0

        # 遍历包含圆形的矩形区域
        for y in range(max(0, cy - radius), min(h, cy + radius + 1)):
            for x in range(max(0, cx - radius), min(w, cx + radius + 1)):
                # 计算点到中心的距离
                distance = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

                # 如果点在圆形区域内
                if distance <= radius:
                    depth_value = depth_map[y, x]
                    # 过滤有效深度值并更新最大值
                    if depth_value > 0 and depth_value > max_depth:
                        max_depth = depth_value

        return float(max_depth)

    def visualize_region(self, image, center_pixel, radius=20, color=(0, 0, 255)):
        # 如果 image 是灰度图，将其转为 BGR
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        img_vis = image.copy()
        cx, cy = center_pixel
        cv2.circle(img_vis, (cx, cy), radius, color, 2)  # 画圆形区域
        cv2.circle(img_vis, (cx, cy), 3, (0, 255, 0), -1)  # 画中心点
        return img_vis


if __name__ == "__main__":
    time_init = time.time()
    sam2 = sam2_auto_mask()
    time_begin = time.time()
    print("sam2_auto_mask init time:", time_begin - time_init)
    
    sam2.logger.info(f"初始化 + load 模型用时:{time_begin - time_init} s ")

    # res , info = sam2.detect_gripper_object()
    # sam2.robot.waypoint(vel=300)
    # breakpoint()
    
    sam2.grasp_count = 0
    
    grasp_size_m = (140, 80) ## 夹爪实际尺寸

    while True:
        sam2.robot.ready(vel=300)
        sam2.robot.open_gripper(2, wait=True)
        grasp_res = True
        print("self.grasp_count:", sam2.grasp_count)
        ready_time = time.time()
        print("ready time:", ready_time - time_begin) #0.05
        
        sam2.logger.info("grasp start:")

        color_image, depth_image = sam2.camera.get_rgbd_image()
        time_before_mask = time.time()
        print("get rgbd time:", time_before_mask - ready_time)  # 0.05
        sam2.logger.info(f"get original rgbd :{time_before_mask - ready_time:.4f} s ")
        
        valid_mask = sam2.get_valid_mask(color_image, depth_image, sam2.grasp_count)
        time_before_grasp = time.time()
        sam2.logger.info(f"original rgbd  --> valid mask:{time_before_mask - time_before_mask:.4f} s")
        print("get valid mask_time:", time_before_grasp - time_before_mask)  # 0.5

        if len(valid_mask) > 0:
            center_pixel = valid_mask[0]["center"]
            vis_img, angle = sam2.visualize_grasps(
                valid_mask[0]["mask"],
                valid_mask[0]["center_depth"],
                color_image,
                center_pixel,
                grasp_size_m,
                sam2.camera_intrinsics,
            )
            if sam2.DEBUG:
                cv2.imshow("Grasp Angle Visualization", vis_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            depth_map = np.array(depth_image)
            # breakpoint()
            depth_map = cv2.resize(
                depth_map, (color_image.shape[1], color_image.shape[0])
            )
            image_np = np.array(color_image)

            # depth_value = depth_map[center_pixel[1], center_pixel[0]]  # 注意坐标顺序

            depth_value = sam2.max_depth_in_radius(depth_map, center_pixel, radius=20)
            # breakpoint()
            depth_value = min(depth_value, 655)

            # 深度值筛选可视化
            # vis = sam2.visualize_region(color_image, center_pixel, radius=20)
            # if sam2.DEBUG:
            #     cv2.imshow("Grasp Angle Visualization", vis)
            #     cv2.waitKey(0)
            #     cv2.destroyAllWindows()

            # depth_value = valid_mask[0]['center_depth']

            # 转换为世界坐标
            target_world_coords = sam2.pixel_to_world_coordinate(
                center_pixel[0], center_pixel[1], depth_value
            )

            # target_world_coords = ready_pose_camera_to_base @ target_world_coords

            suction_position = [
                target_world_coords[0],
                target_world_coords[1],
                target_world_coords[2],
                180,
                0,
                -angle,
            ]
            valid_mask_filter = time.time()
            print(f"get 3D pose time:{valid_mask_filter - time_before_grasp}")  # 0.004
            sam2.robot.arm.set_position(x=120, relative=True, speed=300, wait=True)
            
            # sam2.logger.info(f"original rgbd  --> valid mask:{time_before_mask - time_before_mask:.4f} s")
            sam2.logger.info(f"valid mask filter & waypoint 用时：{valid_mask_filter - time_before_mask} s")

            ## 判断是否两次都抓到相同位置 如果相同 则不加入扰动策略 
            if sam2.pre_grasp_position is not None:
                if sam2.pre_grasp_check(suction_position):
                    sam2.pre_grasp_change(suction_position)
                    time.sleep(0.1)
                    # grasp_res, info = sam2.detect_gripper_object()
                    if grasp_res:
                        sam2.robot.throw(vel=300)
                        sam2.robot.open_gripper(2)
                        sam2.robot.waypoint(vel=300)
                        print("grasp success")
                        sam2.pre_grasp_success = True
                    else:
                        sam2.robot.arm.set_position(
                            x=int(np.random.randint(1, 5) * 10),
                            y=int(np.random.randint(-2, 3) * 10),
                            relative=True,
                            speed=200,
                            wait=True,
                        )
                        sam2.robot.arm.set_position(
                            yaw=int(np.random.randint(0, 6) * 10),
                            relative=True,
                            speed=200,
                            wait=True,
                        )
                        sam2.robot.open_gripper(2)
                    continue
            
            # 开始抓取 
            try:
                sam2.pre_grasp_position = suction_position
                sam2.grasp(suction_position)
            except Exception as e:
                sam2.logger.warning(f"grasp fail 抓取过程中报错 自动重启 {e}")  
                print(f"grasp fail 抓取过程中报错 自动重启 {e}")
                sam2.robot.arm.clean_error()
                sam2.robot.arm.motion_enable(True)
                sam2.robot.arm.set_state(state=0)
                sam2.robot.open_gripper(0)
                sam2.robot.waypoint()
                sam2.pre_grasp_success = True
                continue
            
            time_grasp = time.time()
            sam2.logger.info(f"grasp执行时间{time_grasp - time_before_grasp}")  
            print("time_grasp:", time_grasp - time_before_grasp)  # 5.27

            sam2.robot.waypoint(vel=300)
            # time.sleep(0.1)
            

            # 检查是否抓取成功 
            # grasp_res, info = sam2.detect_gripper_object()
            time_detect = time.time()
            sam2.logger.info(f"检查grasp是否抓取成功 用时：{time_detect - time_grasp}")
            print("time_detect:", time_detect - time_grasp)  # 2.9 -> 1.8


            if grasp_res:
                before_throw_time = time.time()
                sam2.logger.success(
                    f"grasp success 总用时：{before_throw_time - time_before_mask} s"
                )
                sam2.robot.throw(vel=300)
                sam2.robot.open_gripper(2)
                sam2.robot.waypoint(vel=300)
                print("grasp success")
                after_throw_time = time.time()
                sam2.logger.success(
                    f"throw 用时：{after_throw_time - before_throw_time} s"
                )
            else:
                throw_time = time.time()
                sam2.logger.critical(
                    f"grasp fail 总用时：{throw_time - time_before_mask}"
                )
                print("grasp failed")

            sam2.pre_grasp_success = grasp_res

        sam2.grasp_count += 1
        sam2.logger = sam2.logger.bind(grasp_count=sam2.grasp_count)

        # self.close_gripper(2, wait=True)