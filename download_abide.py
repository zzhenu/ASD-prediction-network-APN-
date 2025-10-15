#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data download for selected subjects

Usage:
  download_abide.py [--pipeline=cpac] [--strategy=filt_global] [<derivative> ...]
  download_abide.py (-h | --help)

Options:
  -h --help              Show this screen
  --pipeline=cpac        Pipeline [default: cpac]
  --strategy=filt_global Strategy [default: filt_global]
  derivative             Derivatives to download
"""

# 1. 导入模块
import os
import urllib
import urllib.request
from docopt import docopt

# 2. 定义 collect_and_download 函数
# 收集和下载指定类型的文件。
"""
derivative: 数据类型（如rois_aal, rois_cc200等）。
pipeline: 数据处理管道（如cpac）。
strategy: 数据处理策略（如filt_global）。
out_dir: 下载文件的输出目录。
subject_ids: 需要下载的受试者ID列表。
"""


def collect_and_download(derivative, pipeline, strategy, out_dir, subject_ids):
    # 3. 设置S3前缀
    s3_prefix = "https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative"

    # 4. 处理输入参数
    derivative = derivative.lower()
    pipeline = pipeline.lower()
    strategy = strategy.lower()

    # 5. 确定文件扩展名
    if "roi" in derivative:
        extension = ".1D"
    else:
        extension = ".nii.gz"

    # 6. 创建输出目录
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # 7. 打开并读取表型文件，表型文件包含每个样本的文件ID和其他信息
    s3_pheno_file = open("./data/phenotypes/Phenotypic_V1_0b_preprocessed1.csv", "r")
    pheno_list = s3_pheno_file.readlines()

    # 8. 解析表型文件头：解析表型文件的头信息，找到SUB_ID列的索引
    header = pheno_list[0].split(",")
    file_idx = header.index("FILE_ID")
    sub_id_idx = header.index("SUB_ID")  # 查找SUB_ID列的索引

    # 9. 构建S3路径列表
    s3_paths = []
    for pheno_row in pheno_list[1:]:
        cs_row = pheno_row.split(",")
        sub_id = cs_row[sub_id_idx]  # 提取SUB_ID
        row_file_id = cs_row[file_idx]
        print(sub_id)
        if row_file_id == "no_filename":  # 若某行的 FILE_ID 为 "no_filename"，则跳过该行。
            continue

        # 只下载在 subject_ids 中的受试者数据
        if sub_id not in subject_ids:
            continue

        filename = row_file_id + "_" + derivative + extension
        s3_path = "/".join([s3_prefix, "Outputs", pipeline, strategy, derivative, filename])
        s3_paths.append(s3_path)

    # 10. 下载文件
    total_num_files = len(s3_paths)
    for path_idx, s3_path in enumerate(s3_paths):
        rel_path = s3_path.lstrip(s3_prefix).split("/")[-1]
        download_file = os.path.join(out_dir, rel_path)
        download_dir = os.path.dirname(download_file)
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)
        if not os.path.exists(download_file):
            print("Retrieving: %s" % download_file)
            urllib.request.urlretrieve(s3_path, download_file)
            print("%.3f%% percent complete" % (100 * (float(path_idx + 1) / total_num_files)))
        else:
            print("File %s already exists, skipping..." % download_file)


# 11. 主程序入口
if __name__ == "__main__":

    # 12. 解析命令行参数
    arguments = docopt(__doc__)

    # 13. 设置默认数据类型
    # if not arguments['<derivative>']:
    #     arguments['<derivative>'] = ['rois_aal', 'rois_cc200', 'rois_cc400','rois_dosenbach160', 'rois_ez', 'rois_ho', 'rois_tt']
        
    if not arguments['<derivative>']:
        arguments['<derivative>'] = [
             'rois_cc400'
            
        ]

    # 14. 设置默认管道和策略
    pipeline = arguments.get('pipeline', 'cpac')
    strategy = arguments.get('strategy', 'filt_global')

    # 15. 设置输出目录的绝对路径
    out_dir = os.path.abspath("data/functionals/cpac/filt_global/")

    # 16. 读取 subject.txt 文件，获取受试者ID列表
    subject_file = './subject_IDs.txt'  # 直接在此处指定subject.txt文件的路径
    with open(subject_file, 'r') as f:
        subject_ids = [line.strip() for line in f.readlines()]

    # 17. 遍历数据类型并下载文件
    for derivative in arguments['<derivative>']:
        collect_and_download(derivative, pipeline, strategy, os.path.join(out_dir, derivative), subject_ids)
