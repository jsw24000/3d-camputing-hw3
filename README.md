# 3D Scene Reconstruction Homework 3

本仓库是三维计算成像基础作业三的实验代码与结果整理。场景为华为手机绕教室讲台一圈拍摄的室内讲台区域，实验比较 COLMAP、Nerfacto/NeRF、官方 3DGS 和 VGGT-GS。

## 环境说明

主要在 Windows 11 + RTX 4060 Laptop 8GB 上运行；VGGT 重建部分在 AutoDL 的 RTX 4090 48GB 上运行。

本地主要环境：

```powershell
conda activate torch_env
```

另外使用过：

- `scene-nerf`：运行 Nerfstudio / Nerfacto。
- `scene-3dgs`：运行 GraphDeCo 官方 3D Gaussian Splatting。
- COLMAP：用于相机位姿和稀疏点云恢复。

具体路径和参数见 `config/project.yaml`。

## 数据路径

- 原始视频：`data/raw/desk.mp4`
- 抽帧图像：`data/images/`
- 训练图像：`data/images_train/`
- 测试图像：`data/images_test/`
- 训练/测试划分：`data/splits/train.txt`、`data/splits/test.txt`
- Nerfstudio 输入：`data/nerfstudio_colmap_train/`
- 3DGS 输入：`data/3dgs_full_undistorted/`
- VGGT-GS 输入：`data/3dgs_vggt_51/`

本次使用 51 张图像，其中 46 张训练、5 张测试。

## 主要复现实验命令

数据准备与 COLMAP：

```powershell
python scripts/inspect_video_camera.py
python scripts/prepare_video.py
python scripts/split_dataset.py
python scripts/run_colmap.py
python scripts/analyze_colmap_geometry.py
```

Nerfacto / NeRF：

```powershell
conda run -n scene-nerf python scripts/run_experiment_suite.py --method nerfstudio --stage train --env-mode fallback
python scripts/make_nerfstudio_compat_testset.py
```

官方 3DGS：

```powershell
python scripts/run_3dgs.py --mode all --source-mode full --profile full --eval --llffhold 0
```

VGGT-GS：

```powershell
python scripts/monitor_run.py --method vggt --stage reconstruction --interval 1 -- python scripts/run_vggt_gs.py --mode vggt
python scripts/run_3dgs.py --mode all --source-mode vggt --profile full --eval --llffhold 0
```

指标汇总：

```powershell
python scripts/evaluate_all.py --method all
python scripts/collect_results.py
```

## 输出位置

- COLMAP 几何结果：`results/colmap/`
- Nerfacto 渲染与评估：`results/nerfstudio/`
- 3DGS 模型与渲染：`results/3dgs/full_model/`
- VGGT-GS 重建、模型与渲染：`results/vggt_gs/`
- 定量指标：`results/metrics.csv`
- 时间记录：`results/timing.csv`
- 日志：`logs/`
- 报告图片：`report/assets/`
- 最终报告：`report/report.pdf`

## 提交包结构建议

按照作业建议的结构，当前仓库内容可整理为：

```text
student_id_name_assignment/
  report.pdf              # 对应 report/report.pdf
  README.md               # 本文件
  configs/                # 对应 config/project.yaml
  scripts/                # 实验脚本
  results/
    metrics.csv           # 对应 results/metrics.csv
    timing.csv            # 对应 results/timing.csv
    qualitative/          # 渲染对比图、失败案例图、报告可视化图片
    geometry/             # COLMAP/VGGT 点云、相机轨迹、几何摘要
  logs/                   # 关键运行日志
```

建议放入 `qualitative/` 的文件包括报告中的对齐渲染图、Nerfacto 失败案例图和 3DGS viewer 截图；建议放入 `geometry/` 的文件包括 COLMAP 轨迹图、COLMAP/VGGT 点云图和几何摘要文件。
