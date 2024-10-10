import os
import json
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import numpy as np
from PIL import Image

# 初始化SAM2预测器
checkpoint = "./checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

# 图像和bbox目录路径
image_base_path = "/data/WebDAV/VidOR/video_frames_extract"  #图像帧路径
bbox_base_path = "/data/WebDAV/VidOR/all_training_bbox"  #bbox的 路径
output_mask_path = "/data/WebDAV/VidOR/mask_final" #
output_mask_path_initial = "/data/WebDAV/VidOR/mask_initial" #box作为prompt输入SAM2的结果

# 创建保存mask的文件夹
os.makedirs(output_mask_path, exist_ok=True)
os.makedirs(output_mask_path_initial, exist_ok=True)

# # 从bbox区域采样点（这里随机采样一些点作为示例）
# def sample_points_from_bbox(bbox, num_points=5):
#     xmin, ymin, xmax, ymax = bbox
#     x_coords = np.random.randint(xmin, xmax, size=num_points)
#     y_coords = np.random.randint(ymin, ymax, size=num_points)
#     return np.column_stack((x_coords, y_coords))

# 从生成的掩码中采样正点
def sample_points_from_mask(mask, num_points=5):
    # 获取mask中所有正点的位置 (非零点)
    y_coords, x_coords = np.where(mask > 0)
    # 如果正点数量不足，取所有正点
    if len(x_coords) < num_points:
        sampled_indices = np.arange(len(x_coords))
    else:
        sampled_indices = np.random.choice(len(x_coords), num_points, replace=False)
    sampled_points = np.column_stack((x_coords[sampled_indices], y_coords[sampled_indices]))
    return sampled_points

# 遍历video_frames_extract目录中的所有子文件夹
for video_segment in os.listdir(image_base_path):
    # 提取video_id和帧信息
    video_id, start_frame, end_frame = video_segment.split('_')

    if int(start_frame) == 0:
        start_frame = str(int(start_frame) + 1)

    # 获取视频段的第一帧路径
    first_frame_path = os.path.join(image_base_path, video_segment, "{:04d}.png".format(int(start_frame)))

    if not os.path.exists(first_frame_path):
        print(f"帧 {first_frame_path} 不存在，跳过该视频段.")
        continue

    # 打开第一帧图像
    image = Image.open(first_frame_path).convert("RGB")
    image_array = np.array(image)

    # 找到对应的bbox json文件
    json_file_path = os.path.join(bbox_base_path, f"{video_id}.json")
    if not os.path.exists(json_file_path):
        print(f"对应的bbox文件 {json_file_path} 不存在，跳过该视频段.")
        continue

    # 读取json文件获取bbox
    with open(json_file_path, "r") as f:
        data = json.load(f)

    # 查找对应开始帧的bbox（假设trajectories中的索引与帧一一对应）
    frame_index = int(start_frame) - 1  # 帧索引从0开始，所以减去1
    if frame_index >= len(data['trajectories']):
        print(f"帧 {start_frame} 超出 {video_id} 的轨迹范围.")
        continue

    frame_bboxes = data['trajectories'][frame_index]

    # 对每个object生成mask
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        predictor.set_image(image_array)

        for obj in frame_bboxes:
            tid = obj['tid']
            bbox = obj['bbox']

            # Step 1: 使用bbox生成初步的SAM2 mask
            bbox_array = np.array([bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']])
            masks, _, _ = predictor.predict(box=bbox_array, multimask_output=False)
            # # 转换bbox为np数组
            # bbox_array = np.array([bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']])

            # 将mask转换为uint8类型
            initial_mask = (masks[0] * 255).astype(np.uint8)

            # 保存初步的mask
            initial_mask_pil = Image.fromarray(initial_mask)
            initial_mask_filename = os.path.join(output_mask_path_initial, f"{video_id}_{start_frame}_{end_frame}_{tid}.png")
            initial_mask_pil.save(initial_mask_filename)
            # print(f"保存初步的mask: {initial_mask_filename}")

            # Step 2: 从生成的初步mask中采样正点
            sampled_points = sample_points_from_mask(initial_mask)
            point_labels = np.ones(len(sampled_points))  # 所有采样点都是正点

            # Step 3: 使用采样点、mask和bbox作为prompt，再次输入SAM2进行掩码生成
            final_masks, _, _ = predictor.predict(
                # mask_input=masks,
                box=bbox_array,
                point_coords=sampled_points,
                point_labels=point_labels,
                multimask_output=False
            )
            # # 从bbox区域采样正点
            # point_coords = sample_points_from_bbox(bbox_array, num_points=5)
            # point_labels = np.ones(len(point_coords))  # 所有点都标记为正点（标签为1）

            # # 生成mask
            # masks, _, _ = predictor.predict(box=bbox_array, multimask_output=False)

            # 将最终生成的mask保存
            final_mask = (final_masks[0] * 255).astype(np.uint8)
            final_mask_pil = Image.fromarray(final_mask)
            final_mask_filename = os.path.join(output_mask_path, f"{video_id}_{start_frame}_{end_frame}_{tid}.png")
            final_mask_pil.save(final_mask_filename)
            # print(f"保存最终的mask: {final_mask_filename}")
