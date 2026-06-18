# Experiment Commands

Current data, COLMAP poses, training inputs, and evaluation entry points are prepared. Install external tools following `INSTALL_EXTERNAL.md` before neural method training.

## 0. Status

```powershell
conda run -n torch_env python scripts/check_env.py --dry-run
conda run -n torch_env python scripts/inspect_video_camera.py
conda run -n torch_env python scripts/write_method_manifest.py
conda run -n torch_env python scripts/check_readiness.py
conda run -n torch_env python scripts/pipeline.py status
```

## 1. Data And COLMAP

Rebuild the current preparation pipeline:

```powershell
conda run -n torch_env python scripts/pipeline.py prepare-all
conda run -n torch_env python scripts/run_experiment_suite.py --stage prepare
```

Preview only:

```powershell
conda run -n torch_env python scripts/pipeline.py prepare-all --dry-run
conda run -n torch_env python scripts/run_experiment_suite.py --stage prepare --dry-run
```

Expected COLMAP outputs:

```text
results/colmap/sparse_point_cloud.ply
results/colmap/train_sparse_point_cloud.ply
results/colmap/train_geometry_summary.json
results/colmap/train_geometry_preview.png
data/3dgs_full_undistorted/
data/3dgs_train_undistorted/
```

## 2. Nerfacto / NeRF

Preview architecture and commands. The adapter runs from `torch_env` and calls Nerfstudio commands in `scene-nerf` by default:

```powershell
conda run --no-capture-output -n torch_env python scripts/run_nerfstudio.py --mode train --profile smoke --dry-run
conda run --no-capture-output -n torch_env python scripts/run_nerfstudio.py --mode eval --dry-run
conda run --no-capture-output -n torch_env python scripts/run_nerfstudio.py --mode render --dry-run
```

Smoke train after `scene-nerf` has Nerfstudio installed:

```powershell
conda run --no-capture-output -n torch_env python scripts/run_nerfstudio.py --mode train --profile smoke
conda run --no-capture-output -n torch_env python scripts/run_nerfstudio.py --mode eval
```

Full train:

```powershell
conda run --no-capture-output -n torch_env python scripts/run_nerfstudio.py --mode train --profile full
conda run --no-capture-output -n torch_env python scripts/run_nerfstudio.py --mode eval
```

Unified runner:

```powershell
conda run --no-capture-output -n torch_env python scripts/run_experiment_suite.py --method nerfstudio --stage train --nerf-profile smoke --dry-run
conda run --no-capture-output -n torch_env python scripts/run_experiment_suite.py --method nerfstudio --stage train --nerf-profile full
conda run --no-capture-output -n torch_env python scripts/run_experiment_suite.py --method nerfstudio --stage eval
```

## 3. Official 3DGS

Prepare undistorted PINHOLE/SIMPLE_PINHOLE datasets for GraphDeCo 3DGS:

```powershell
conda run -n torch_env python scripts/prepare_3dgs_undistorted.py
```

Preview smoke commands. `run_3dgs.py` defaults to the `scene-3dgs` Conda env for official training/rendering:

```powershell
conda run --no-capture-output -n torch_env python scripts/run_3dgs.py --mode all --source-mode full --profile smoke --dry-run
```

Smoke train and render:

```powershell
conda run --no-capture-output -n torch_env python scripts/run_3dgs.py --mode train --source-mode full --profile smoke
conda run --no-capture-output -n torch_env python scripts/run_3dgs.py --mode render --source-mode full --profile smoke
```

Quick qualitative lectern model:

```powershell
conda run --no-capture-output -n torch_env python scripts/run_3dgs.py --mode train --source-mode full --profile quick
conda run --no-capture-output -n torch_env python scripts/run_3dgs.py --mode render --source-mode full --profile quick
```

Full qualitative lectern model:

```powershell
conda run --no-capture-output -n torch_env python scripts/timed_run.py --method 3dgs --stage train -- python scripts/run_3dgs.py --mode train --source-mode full --profile full
conda run --no-capture-output -n torch_env python scripts/timed_run.py --method 3dgs --stage render -- python scripts/run_3dgs.py --mode render --source-mode full --profile full
```

Strict train-only model for later hold-out metrics:

```powershell
conda run -n torch_env python scripts/run_3dgs.py --mode train --source-mode train --profile full
conda run -n torch_env python scripts/run_3dgs.py --mode render --source-mode train --profile full
```

Key 3DGS outputs:

```text
results/3dgs/eval_model/test/ours_30000/renders/
results/3dgs/eval_model/test/ours_30000/gt/
results/3dgs/full_model/point_cloud/iteration_30000/point_cloud.ply
results/3dgs/full_model/train/ours_30000/renders/
```

## 4. VGGT-GS

Preview:

```powershell
conda run -n torch_env python scripts/run_vggt_gs.py --dry-run
```

Run VGGT export and GS entry point:

```powershell
conda run -n scene-vggt python scripts/timed_run.py --method vggt_gs --stage all -- python scripts/run_vggt_gs.py --mode all
conda run -n torch_env python scripts/run_experiment_suite.py --method vggt_gs --stage train --env-mode fallback
```

## 5. Evaluation And Collection

Preview evaluation directories:

```powershell
conda run -n torch_env python scripts/evaluate_all.py --dry-run
conda run -n torch_env python scripts/run_experiment_suite.py --stage eval --dry-run
```

After training and rendering:

```powershell
conda run -n torch_env python scripts/evaluate_all.py
conda run -n torch_env python scripts/collect_results.py
conda run -n torch_env python scripts/run_experiment_suite.py --stage collect
```

## 6. Full Preview

Preview all train, render, evaluation, and collection commands:

```powershell
conda run -n torch_env python scripts/run_experiment_suite.py --method all --stage all --dry-run
```

Run the full pipeline after external dependencies and fallback envs are installed:

```powershell
conda run -n torch_env python scripts/run_experiment_suite.py --method all --stage all --env-mode fallback
```

Main outputs:

```text
results/metrics.csv
results/timing.csv
results/model_sizes.csv
results/summary.json
```
