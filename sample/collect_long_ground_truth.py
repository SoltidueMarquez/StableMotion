"""
从指定的 AMASS 文件夹采集完整长度的损坏动作，用作长序列修复的 GT。
本脚本等价于 `sample.fix_globsmpl` 中的 collect 模式，但明确切换到 folder/sequential
模式，因此每条 clip 都会按原始长度读取并保存（不会裁剪到 100 帧）。
保存结果可供长序列评估脚本使用。
"""

import os
from argparse import ArgumentParser

import numpy as np
from tqdm import tqdm

from data_loaders.amasstools.globsmplrifke_feats import globsmplrifkefeats_to_smpldata
from data_loaders.get_data import get_dataset_loader
from utils.parser_util import add_data_options


def parse_args():
    parser = ArgumentParser(description="收集长序列的 Ground Truth 动作")
    add_data_options(parser)
    parser.add_argument(
        "--folder_path",
        type=str,
        default="",
        help="遍历指定的 motion 文件夹，按整段顺序读取（required for long GT）。",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="DataLoader batch size.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="是否在遍历 folder_path 时打乱顺序（默认不打乱）。",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output/long_ground_truth",
        help="保存结果的目录。",
    )
    parser.add_argument(
        "--file_name",
        type=str,
        default="long_ground_truth.npy",
        help="输出 .npy 文件的基本名称。",
    )
    parser.add_argument(
        "--ext",
        type=str,
        default="",
        help="可选：在输出文件名后追加后缀用于区分不同配置。",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.folder_path:
        raise ValueError("请通过 --folder_path 指定包含 motion 的文件夹。")

    data = get_dataset_loader(
        name="globsmpl",
        batch_size=args.batch_size,
        split="test_benchmark",
        shuffle=args.shuffle,
        sample_mode="sequential",
        folder_path=args.folder_path,
        data_dir=args.data_dir,
        normalizer_dir=args.normalizer_dir,
    )
    motion_normalizer = data.dataset.motion_loader.motion_normalizer

    # 预分配采集结果容器
    all_gt_motions = []
    all_gt_labels = []
    all_lengths = []
    all_input_features = []
    total_samples = 0

    # 遍历 DataLoader 的每个 batch，batch 中的 clip 都是整段（无随机裁剪）
    for batch in tqdm(data, desc="Collecting long GT"):
        input_motions = batch["x"]
        lengths = batch["length"]
        # 反归一化到物理尺度，便于下一步拆出标签/动作
        temp_sample = motion_normalizer.inverse(input_motions.transpose(1, 2).cpu())

        batch_size = temp_sample.shape[0]
        for idx in range(batch_size):
            sample = temp_sample[idx]
            all_gt_labels.append((sample[..., -1] > 0.5).numpy())  # GT 标签通道
            all_gt_motions.append(globsmplrifkefeats_to_smpldata(sample[..., :-1]))  # 转 SMPL dict
            all_input_features.append(sample.numpy())  # 保存完整归一化特征（可供后续 debug）
        all_lengths.extend(lengths.cpu().numpy().tolist())
        total_samples += batch_size

    os.makedirs(args.output_dir, exist_ok=True)
    file_suffix = f"_{args.ext}" if args.ext else ""
    output_name = args.file_name.replace(".npy", f"{file_suffix}.npy")
    output_path = os.path.join(args.output_dir, output_name)

    print(f"保存 GT ({total_samples} 个 clip) 到 {output_path}")
    np.save(
        output_path,
        {
            "motion": all_gt_motions,
            "gt_labels": all_gt_labels,
            "lengths": np.array(all_lengths, dtype=np.int32),
            "input_features": all_input_features,
        },
    )
    length_txt = output_path.replace(".npy", "_len.txt")
    with open(length_txt, "w", encoding="utf-8") as fw:
        fw.write("\n".join(str(l) for l in all_lengths))


if __name__ == "__main__":
    main()

