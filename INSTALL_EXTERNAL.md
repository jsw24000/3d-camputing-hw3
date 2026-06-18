# External Training Dependencies

当前仓库的数据预处理、COLMAP、训练输入准备都已经完成。真正训练 NeRF、3DGS、VGGT-GS 前，还需要安装外部工具。

## 当前机器状态

`torch_env` 已经有 CUDA 版 PyTorch：

- PyTorch: `2.6.0+cu124`
- GPU: RTX 4060 Laptop GPU
- Python: `3.13.2`

但以下工具缺失：

- `nerfstudio`
- official `gaussian-splatting`
- `VGGT`
- `gsplat`
- `lpips`

## 推荐策略

不要把所有依赖强塞进一个环境。如果 `torch_env` 的 Python 3.13 导致安装失败，优先拆分环境：

- `scene-nerf`：nerfstudio / Nerfacto
- `scene-3dgs`：官方 GraphDeCo 3DGS
- `scene-vggt`：VGGT + gsplat

项目内脚本和数据路径保持不变。

## Nerfstudio

官方文档说明 Windows 安装更脆弱，推荐 Linux 或 WSL2；官方示例环境使用较旧 Python 和 CUDA 组合。当前 `torch_env` 是 Python 3.13，因此如果直接安装失败，不要硬修 `torch_env`，新建 `scene-nerf` 更稳。

建议先尝试：

```powershell
conda create -n scene-nerf python=3.10 -y
conda activate scene-nerf
python -m pip install --upgrade pip
pip install nerfstudio
```

训练命令：

```powershell
conda run -n scene-nerf python scripts/timed_run.py --method nerfstudio --stage train -- python scripts/run_nerfstudio.py --mode train
```

## Official 3DGS

官方 3DGS README 的基础训练形式是：

```powershell
python train.py -s <path to COLMAP or NeRF Synthetic dataset>
```

本项目已经准备好 COLMAP dataset：

```text
data/3dgs_train/
  images/
  sparse/0/
```

建议目录：

```powershell
git clone https://github.com/graphdeco-inria/gaussian-splatting external/gaussian-splatting
```

官方仓库有 `environment.yml`，建议按官方环境创建；如果 Windows 编译 CUDA 扩展失败，优先换 WSL2。

训练命令：

```powershell
conda run -n scene-3dgs python scripts/timed_run.py --method 3dgs --stage train -- python scripts/run_3dgs.py --mode train
```

## VGGT + GS

VGGT 官方支持导出 COLMAP 格式：

```powershell
python demo_colmap.py --scene_dir=/YOUR/SCENE_DIR/
python demo_colmap.py --scene_dir=/YOUR/SCENE_DIR/ --use_ba --max_query_pts=2048 --query_frame_num=5
```

官方要求图像放在：

```text
SCENE_DIR/images/
```

本项目脚本会把 `data/images_train/` 同步到：

```text
data/vggt_scene_train/images/
```

VGGT README 还给出了接 gsplat 的示例：

```powershell
python examples/simple_trainer.py default --data_factor 1 --data_dir /YOUR/SCENE_DIR/ --result_dir /YOUR/RESULT_DIR/
```

本项目入口：

```powershell
conda run -n scene-vggt python scripts/run_vggt_gs.py --dry-run
```

去掉 `--dry-run` 前，需要先安装 VGGT 和 gsplat。

## LPIPS

LPIPS 只用于指标评估，失败时不影响 PSNR/SSIM：

```powershell
pip install lpips
```

如果安装失败，`scripts/evaluate_render.py` 会写入 `LPIPS=unavailable`。

## Sources

- Nerfstudio installation: https://docs.nerf.studio/quickstart/installation.html
- GraphDeCo 3DGS: https://github.com/graphdeco-inria/gaussian-splatting
- VGGT: https://github.com/facebookresearch/vggt
