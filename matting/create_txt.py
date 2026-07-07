#!/usr/bin/env python3
"""
生成 filenames_train.txt 用于训练
每行包含两列: Image路径(kontext img) GT路径 (用空格分隔)
trimap由训练代码从alpha自动生成，无需提供
"""

import os

# 目录配置
IMAGE_DIR = "/nvmedata/workspace2/users/rzc/datasets/merged_matting_dataset/train/original"  # kontext img
GT_DIR = "/nvmedata/workspace2/users/rzc/datasets/merged_matting_dataset/train/mask"
OUTPUT_FILE = "/nvmedata/workspace2/users/rzc/datasets/merged_matting_dataset/train/filenames_train.txt"


def get_sorted_files(directory, extensions=None):
    """获取目录中的所有文件并排序"""
    if not os.path.exists(directory):
        print(f"错误: 目录不存在 - {directory}")
        return []

    files = []
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        if os.path.isfile(file_path):
            if extensions is None or any(file.lower().endswith(ext) for ext in extensions):
                files.append(file_path)

    return sorted(files)


def create_filenames_txt():
    """创建 filenames_train.txt 文件"""

    print("=" * 60)
    print("生成 filenames_train.txt")
    print("=" * 60)
    print()

    # 检查目录
    print("检查目录...")
    if not os.path.exists(IMAGE_DIR):
        print(f"✗ Image (kontext img) 目录不存在: {IMAGE_DIR}")
        return False
    print(f"✓ Image (kontext img) 目录: {IMAGE_DIR}")

    if not os.path.exists(GT_DIR):
        print(f"✗ GT 目录不存在: {GT_DIR}")
        return False
    print(f"✓ GT 目录: {GT_DIR}")
    print()

    # 获取文件列表
    print("扫描文件...")
    image_files = get_sorted_files(IMAGE_DIR, extensions=['.png', '.jpg', '.jpeg'])
    gt_files = get_sorted_files(GT_DIR, extensions=['.png', '.jpg', '.jpeg'])

    print(f"Image (kontext img) 文件数量: {len(image_files)}")
    print(f"GT 文件数量: {len(gt_files)}")
    print()

    if len(image_files) == 0:
        print("✗ 错误: Image 目录中没有图像文件")
        return False

    if len(gt_files) == 0:
        print("✗ 错误: GT 目录中没有图像文件")
        return False

    # 创建文件名映射（去掉扩展名）
    image_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in image_files}
    gt_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in gt_files}

    # 找到匹配的文件对
    print("匹配文件对...")
    matched_pairs = []
    unmatched_image = []
    unmatched_gt = []

    for name in sorted(image_dict.keys()):
        if name in gt_dict:
            matched_pairs.append((image_dict[name], gt_dict[name]))
        else:
            unmatched_image.append(name)

    for name in gt_dict.keys():
        if name not in image_dict:
            unmatched_gt.append(name)

    print(f"✓ 匹配的文件对: {len(matched_pairs)}")

    if unmatched_image:
        print(f"⚠ 警告: {len(unmatched_image)} 个 Image 文件没有对应的 GT")
        if len(unmatched_image) <= 5:
            for name in unmatched_image:
                print(f"    - {name}")

    if unmatched_gt:
        print(f"⚠ 警告: {len(unmatched_gt)} 个 GT 文件没有对应的 Image")
        if len(unmatched_gt) <= 5:
            for name in unmatched_gt:
                print(f"    - {name}")

    if len(matched_pairs) == 0:
        print("✗ 错误: 没有找到匹配的文件对")
        return False

    print()

    # 写入文件
    print(f"写入文件: {OUTPUT_FILE}")
    print("列顺序: Image(kontext_img) GT(alpha/mask)")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for image_path, gt_path in matched_pairs:
            f.write(f"{image_path} {gt_path}\n")

    print(f"✓ 已生成 {OUTPUT_FILE}")
    print()

    # 显示前几行
    print("前5行示例:")
    print("-" * 60)
    with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i < 5:
                print(f"{i+1}: {line.rstrip()}")
            else:
                break
    print("-" * 60)
    print()

    print(f"总共 {len(matched_pairs)} 行")
    print()
    print("=" * 60)
    print("完成!")
    print("=" * 60)

    return True


if __name__ == '__main__':
    try:
        success = create_filenames_txt()
        if not success:
            exit(1)
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        exit(1)