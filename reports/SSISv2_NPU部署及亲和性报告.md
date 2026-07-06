# SSISv2 — 实例阴影检测 NPU 部署及亲和性报告

| 项 | 内容 |
|---|---|
| 任务编号 | SSISv2 |
| 任务用途 | 实例阴影检测: 目标实例、阴影实例与目标-阴影关联预测 |
| 仓库 | SSISv2 |
| 版本 / commit | 4d840f3 / finish |
| 日期 | 2026-07-06 |
| 硬件 | Ascend 950PR / 单卡 NPU 指标口径 / CANN 9.0 / npu-smi 25.7.rc1 |
| 软件 | Python 3.11.6 / torch 2.10.0+cpu / torch_npu 2.10.0 / torchvision 0.25.0+cpu / detectron2 0.6 / numpy 1.26.4 |

---

## 1. 技术栈
- 主语言 Python, 基于 AdelaiDet + Detectron2 的 CondInst/FCOS 单阶段实例分割检测器, 主干为 ResNet101 + BiFPN。
- 任务输出包含 object mask、shadow mask、object-shadow association, 指标包括常规 mask AP / box AP 与项目自定义 SOAP mask / SOAP bbox。
- 项目包含训练入口: README 的 Training 部分使用 `tools/train_net.py --config-file ../configs/MS_R_101_BiFPN_SSISv2.yaml --num-gpus 1`; 配置中 `DATASETS.TRAIN=("soba_cast_shadow_train_full",)`, `SOLVER.MAX_ITER=45000`, `SOLVER.IMS_PER_BATCH=2`。
- CUDA 依赖: 原仓面向 CUDA/Detectron2 扩展; `detectron2._C`、AdelaiDet BezierAlign/DefROIAlign/ModulatedDeformConv、NMS 均有 C++/CUDA 扩展路径。
- 自定义核: `PythonAPI/pysobatools/_mask` Cython 扩展、Detectron2/AdelaiDet C++ 扩展。NPU 部署没有构建 CUDA 扩展。
- 权重: `model_ssisv2_final.pth`, SHA256 `1fed82fb4f79217ea9c730d55c8759a9fdc0c36429965a67dab09c9b1bf004c6`。
- 数据集: `SOBA_v2.zip`, SHA256 `a1a4f2c685636e6d38c4239084905b1ccddf048a217ee7644dd3d1ac67f05b2e`。
- Benchmark: 仓内没有独立 benchmark 脚本。README 给出评测基准; 复现路径为 `tools/train_net.py --eval-only` 生成 AP, 再由 `tools/SOAP.py` 计算 SOAP mask / SOAP bbox。

## 2. NPU 部署改造
- 依赖: 使用 torch_npu 环境; pip cache、临时目录、数据集、权重和输出放在数据盘; detectron2 源码以无 `_C` 扩展方式导入。
- 数据: `SOBA_v2.zip` 展开为 `datasets/SOBA_v2/{annotations,SOBA}`。Challenge 集 1 张 JPEG 的标注尺寸与真实图像宽度不一致, 按图像解码尺寸修正后评测。
- 代码适配:
  - `tools/train_net.py` 引入 `torch_npu`, 三个 SOBA 注册入口改为 `../datasets/SOBA_v2/SOBA`。
  - BezierAlign / DefROIAlign import 加保护; SSISv2 默认路径不依赖这两个 ROI 扩展。
  - `ml_nms` 使用 CPU `batched_nms`, 输出索引转回 NPU, 保证后处理可执行。
  - `ModulatedDeformConv` / `DeformConv` 用普通 `conv2d` 兼容实现替代, 保持权重加载、前向和训练闭环。
  - `pysobatools.mask` 复用 `pycocotools._mask`; `np.float` 改为 `float` 以兼容 numpy 1.26。
  - `aligned_bilinear` 的 replicate padding 改为 NPU-safe 的 slice + expand + cat, 与 `F.pad(..., mode="replicate")` 数值等价, 避开 `replication_pad2d_backward` 在 NPU 上的反向失败。

**运行环境**

```bash
export ASCEND_RT_VISIBLE_DEVICES=<allowed_npu_id>
export WORKSPACE=/path/to/SSISv2
export DETECTRON2_SRC=/path/to/detectron2-src
export CACHE_ROOT=/data/cache/ssisv2
export PYTHONPATH=$WORKSPACE/PythonAPI:$WORKSPACE:$DETECTRON2_SRC:$PYTHONPATH
export TMPDIR=$CACHE_ROOT/tmp
export XDG_CACHE_HOME=$CACHE_ROOT/xdg
export TORCH_HOME=$CACHE_ROOT/torch
```

**训练命令**

```bash
cd tools
python train_net.py \
  --config-file ../configs/MS_R_101_BiFPN_SSISv2.yaml \
  --num-gpus 1 \
  MODEL.DEVICE npu DATALOADER.NUM_WORKERS 0 \
  SOLVER.MAX_ITER 200 SOLVER.CHECKPOINT_PERIOD 200 TEST.EVAL_PERIOD 0 \
  DATASETS.TEST "()" \
  MODEL.WEIGHTS output/SSISv2_MS_R_101_bifpn_with_offset_class_maskiouv2_da_bl/model_ssisv2_final.pth \
  OUTPUT_DIR output/SSISv2_train_path_200iter
```

**推理与指标命令**

```bash
cd tools
python train_net.py \
  --config-file ../configs/MS_R_101_BiFPN_SSISv2.yaml \
  --num-gpus 1 --resume --eval-only \
  MODEL.DEVICE npu DATALOADER.NUM_WORKERS 0

python SOAP.py \
  --path ../datasets/SOBA_v2/annotations/SOBA_challenge_v2.json \
  --input-name ./output/SSISv2_MS_R_101_bifpn_with_offset_class_maskiouv2_da_bl

python train_net.py \
  --config-file ../configs/MS_R_101_BiFPN_SSISv2.yaml \
  --num-gpus 1 --resume --eval-only \
  MODEL.DEVICE npu DATALOADER.NUM_WORKERS 0 \
  DATASETS.TEST "('soba_cast_shadow_val_full',)" \
  OUTPUT_DIR output/SSISv2_MS_R_101_bifpn_with_offset_class_maskiouv2_da_bl_val

python SOAP.py \
  --path ../datasets/SOBA_v2/annotations/SOBA_val_v2.json \
  --input-name ./output/SSISv2_MS_R_101_bifpn_with_offset_class_maskiouv2_da_bl_val
```

## 3. 验证用例
**输入规格**

- 任务输入为 BGR 图片, Detectron2 数据流保持长宽比 resize。
- 训练输入: `INPUT.MIN_SIZE_TRAIN=(640, 672, 704, 736, 768, 800)`, `INPUT.MAX_SIZE_TRAIN=1333`, 每张图随机选择短边尺寸, 长边不超过 1333。
- 推理输入: `INPUT.MIN_SIZE_TEST=800`, `INPUT.MAX_SIZE_TEST=1333`, 短边 resize 到 800, 长边不超过 1333。
- batch: 训练 `IMS_PER_BATCH=2`; 推理 batch=1。
- 数据集原图规格来自实际图像解码与标注 JSON 交叉校验。SOBA-train 840 images / 5998 annotations, 原图分辨率 220x154 到 2304x2168, 面积 38,720 到 3,592,376 像素, 共 412 种分辨率; 高频分辨率为 1024x683(90)、640x480(72)、1024x768(35)、1024x572(31)、645x484(26)、1024x682(25)、1024x576(24)、640x426(15)。
- SOBA-testing 160 images / 1248 annotations, 原图分辨率 297x236 到 2304x1728, 面积 105,732 到 3,981,312 像素, 共 113 种分辨率; 高频分辨率为 1024x683(16)、1024x768(14)、640x480(7)、1280x853(4)、2048x1536(3)、1024x682(2)、768x1024(2)、1024x576(2)。
- SOBA-challenge 100 images / 1340 annotations, 原图分辨率 533x373 到 3888x3888, 面积 271,360 到 10,077,696 像素, 共 78 种分辨率; 高频分辨率为 800x600(9)、2048x1536(5)、799x533(4)、640x480(3)、800x530(2)、533x799(2)、1024x768(2)、1600x1200(2)。
- 三个 split 的标注宽高与实际图像解码尺寸均已一致, missing/error/mismatch 均为 0。Challenge 集 `challenge-shadow079.jpg` 曾存在宽度标注不一致, 以实际图像解码尺寸修正后纳入上述统计。

**训练集与训练闭环**

- SOBA-train: 840 images, 5998 annotations, 3000 association items。
- 200 iter 训练验证使用单卡 NPU、batch=2、FP32 eager, 从已发布 SSISv2 权重继续训练, 目的是验证数据加载、前向、loss、反向、optimizer step 和 checkpoint 写出。官方完整训练配方仍是 45000 iter。
- 训练产物: `metrics.json`, `model_0000199.pth`, `model_final.pth`; 单个 checkpoint 约 515MB。
- `aligned_bilinear` 修正前, 训练在首个 backward 触发 `replication_pad2d_backward` / `PadV3GradAiCore` 失败; 修正后 5 iter、50 iter、200 iter 均正常退出。

| 训练验证 | iter | 总耗时 | 训练速度 | NPU 利用率 | HBM 峰值 | 结果 |
|---|---:|---:|---:|---:|---:|---|
| 单卡训练 | 200 | 126s wall / 111s train hook | 0.5621s/iter | avg 29.03%, max 100% | 25429MB / 114688MB | exit_status=0, checkpoint 写出 |

| iter | total_loss | loss_fcos_cls | loss_fcos_loc | loss_mask | loss_asso_mask | boundary_loss | lr |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 19 | 1.2072 | 0.0656 | 0.1158 | 0.0591 | 0.0679 | 0.2317 | 0.00019081 |
| 39 | 1.1573 | 0.0645 | 0.1085 | 0.0550 | 0.0697 | 0.2432 | 0.00039061 |
| 59 | 1.2676 | 0.0615 | 0.1222 | 0.0744 | 0.0853 | 0.2606 | 0.00059041 |
| 79 | 1.2053 | 0.0640 | 0.1158 | 0.0579 | 0.0678 | 0.2362 | 0.00079021 |
| 99 | 1.2942 | 0.0779 | 0.1471 | 0.0720 | 0.0861 | 0.2682 | 0.00099001 |
| 119 | 1.1444 | 0.0728 | 0.1246 | 0.0534 | 0.0589 | 0.2279 | 0.00100000 |
| 139 | 1.2580 | 0.0685 | 0.1273 | 0.0620 | 0.0682 | 0.2273 | 0.00100000 |
| 159 | 1.2775 | 0.0790 | 0.1391 | 0.0671 | 0.0856 | 0.2653 | 0.00100000 |
| 179 | 1.3036 | 0.0711 | 0.1307 | 0.0697 | 0.0824 | 0.2587 | 0.00100000 |
| 199 | 1.3571 | 0.0835 | 0.1409 | 0.0736 | 0.1136 | 0.3199 | 0.00100000 |

2 进程分布式训练入口也做了受控验证, 只开放允许范围内的 2 张 NPU, `--num-gpus 2`, `MAX_ITER=1`。原生 Detectron2 DDP 失败于 `RuntimeError: No backend type associated with device type npu`; 因此当前可执行训练路径为单卡 NPU。

**评测数据与输出**

- SOBA-challenge: 100 images, 1340 instances。
- SOBA-testing: 160 images, 1248 instances。
- 输出产物均存在: `instances_predictions.pth`, `soba_instances_results.json`, `soba_association_results.json`。

| 数据集 | 指标 | README SSISv2 | Ascend 950PR 实测 | 差值 |
|---|---:|---:|---:|---:|
| SOBA-testing | SOAP mask | 35.3 | 34.4 | -0.9 |
| SOBA-testing | SOAP bbox | 29.0 | 28.9 | -0.1 |
| SOBA-testing | mask AP | 50.2 | 49.137 | -1.063 |
| SOBA-testing | box AP | 44.4 | 44.051 | -0.349 |
| SOBA-challenge | SOAP mask | 17.7 | 16.7 | -1.0 |
| SOBA-challenge | SOAP bbox | 15.1 | 14.8 | -0.3 |
| SOBA-challenge | mask AP | 31.0 | 29.678 | -1.322 |
| SOBA-challenge | box AP | 28.4 | 28.238 | -0.162 |

| 数据集 | 图像数 | Detectron2 总推理 | 纯推理 | 端到端 wall time | 总推理吞吐 |
|---|---:|---:|---:|---:|---:|
| SOBA-testing | 160 | 28.301s / 0.1826s per image | 23s / 0.1487s per image | 40s | 5.65 img/s |
| SOBA-challenge | 100 | 49.958s / 0.5259s per image | 41s / 0.4368s per image | 67s | 2.00 img/s |

与 README 的 CUDA 基准相比, box AP 基本对齐, mask AP 与 SOAP mask 下降约 1 个点。主要差异来自 deformable maskIoU head 的 ModulatedDeformConv 没有原生 NPU/C++ 扩展, 部署中以普通卷积兼容实现替代。

## 4. NPU 亲和性
口径: Ascend 950PR, 架构版本 3510, FP32 eager, 单卡训练 batch=2, 单卡推理 batch=1, 输入短边 800/max 1333。AMP 关闭, 不把 FP16 平衡点直接当作 FP32 评分。该模型是检测/实例分割动态图执行流, 真实瓶颈由卷积主干、动态 mask 分支、NMS、Python evaluator 和训练数据管线共同决定。

NPU 指标统一按单卡任务统计。`ASCEND_RT_VISIBLE_DEVICES=7` 运行验证中, torch_npu 可见设备数为 1, 训练进程在物理卡 7 上分配 HBM, Python 内部设备名为 `npu:0`; 因此训练/推理利用率、HBM 和吞吐均不按多卡汇总。表中 HBM 分母沿用对应运行采样记录的 114688MB; 同型号卡在该环境中 `npu-smi` 显示单卡 HBM 为 131072MB, 不改变已记录任务峰值。

| 指标 | 数值 |
|---|---|
| 能否在 NPU 跑通 | 单卡训练与单卡推理可跑通; 2 进程 DDP 需要 launcher/backend 适配 |
| 训练 NPU 利用率 | 200 iter 训练 1s 采样: avg 29.03%, max 100% |
| 训练 HBM 占用 | 200 iter 训练峰值 25429MB / 114688MB |
| 推理 NPU 利用率 | challenge 评测 1s 采样: avg 11.8%, max 38% |
| 推理 HBM 占用 | challenge 评测峰值 22357MB / 114688MB |
| 关键算子是否回退 CPU | NMS 回 CPU; SOBA/SOAP 评估在 CPU; ModulatedDeformConv 用普通 NPU conv2d 兼容实现 |
| 性能 | 训练 0.5621s/iter; challenge 2.00 img/s; testing 5.65 img/s |

**CPU 口径**

- CPU 参与范围: dataloader/augmentation、CPU `batched_nms`、SOBA/SOAP evaluator、numpy/pycocotools 后处理。
- 任务配置的 CPU 并发: 训练和推理命令均设置 `DATALOADER.NUM_WORKERS=0`, 因此没有额外 dataloader worker 进程; 单卡训练/推理为单主 Python 进程。CPU NMS 与 evaluator 也在主流程中执行。PyTorch/MKL/OpenMP 线程池仍会在主进程下创建工作线程, 不能把 dataloader worker 数等同于 CPU 实际用核数。
- 实际 CPU 核心占用: 对单卡 20 iter 训练进程按 0.2s 间隔采样 `/proc/<pid>/task/*/stat`, 统计有 CPU tick 增量的线程及其调度 CPU。该训练段 `DATALOADER.NUM_WORKERS=0`, `IMS_PER_BATCH=2`, 17 个计时 iteration 平均 0.4987s/iter; 采样到 260 个线程, 全程迁移覆盖 196 个 CPU core, 单个采样窗口最多同时有 132 个活跃线程、130 个活跃 CPU core。CPU 任务用核以“最大同时活跃 CPU core=130”描述, 全程 196 个 core 是调度迁移覆盖范围, 不是同时用核。
- x86 AVX: CPU 架构为 x86_64, 处理器为 AMD EPYC 9575F。`lscpu` flags 包含 AVX、AVX2、AVX512F、AVX512DQ、AVX512BW、AVX512VL、AVX512_BF16、AVX512_VNNI、AVX_VNNI、FMA 等; PyTorch build info 显示 `CPU capability usage: AVX512`, 并启用 MKL、MKL-DNN、OpenMP、`PERF_WITH_AVX=1`、`PERF_WITH_AVX2=1`。这说明 CPU fallback 与后处理可使用 AVX/AVX2/AVX512 优化路径, 但具体到 NMS、evaluator、numpy/pycocotools 每个热点的指令级命中需要 perf/VTune 类工具另行确认。

**计算分布**

| 单元 | 主要算子 / 阶段 | 压力 | 说明 |
|---|---|---:|---|
| 算力(Cube,卷积/线性) | ResNet101、BiFPN、FCOS head、mask branch、maskIoU FC | 中高 | 大部分卷积和线性层在 NPU 上执行; 训练反向使卷积梯度成为主要 NPU 负载 |
| 向量(Vector,归一/激活/损失) | GroupNorm、BatchNorm、ReLU、sigmoid、IoU/loss、mask 后处理 | 中 | 小 tensor、多尺度 feature、动态实例数带来较多 Vector kernel |
| 搬运(MTE/FixPipe) | 多尺度 resize/interpolate、slice/concat、mask logits upsample、padding 等价重写 | 中高 | 训练中 NPU-safe replicate pad 由 slice + expand + cat 组成, 保持语义但增加搬运与 kernel 数 |
| 通信(communication) | 单卡训练/推理 | 低 | 单卡路径无 collective; 原生 2 进程 DDP 在 NPU backend 处失败 |
| 调度(host/head) | Python dataloader、augmentation、polygon/RLE、NMS、SOBA/SOAP evaluator | 高 | 200 iter 训练日志 data_time 平均 0.3743s, logged time 平均 0.5154s; 推理 wall time 明显高于纯推理时间 |

**判断依据**

950PR 属 3510 架构, AIC/AIV 分离, 适合卷积/GEMM 主体走 Cube, elementwise 与归一走 Vector。SSISv2 主干卷积和 FCOS head 算术密度较高, 对 NPU 亲和; 但检测后处理和动态实例分割把大量小 kernel、索引、reshape、mask upsample、CPU evaluator 串在主路径上, 导致平均利用率低于峰值利用率。200 iter 训练峰值利用率达到 100%, 说明 NPU 计算段能吃满; 平均 29.03% 反映 host/data 与动态图小算子间隙占比高。

训练比推理更能暴露 NPU 算子覆盖问题。原 `F.pad(..., mode="replicate")` 前向可执行, 但反向在 `PadV3GradAiCore` 失败; 采用等价的 slice + expand + cat 后, CPU 上与原 replicate pad 最大误差为 0, NPU 上 `aligned_bilinear(...).sum().backward()` 通过, 训练 200 iter 通过。该重写属于数学等价流程, 亲和性从“算子不支持导致硬失败”提升为“可执行但增加 MTE/Vector 与 launch 压力”。

**热点与瓶颈**

- ResNet101 + BiFPN: 亲和中高。卷积主体适合 Cube; BiFPN 多尺度融合带来 interpolate、pool、concat 和小 feature map kernel, 平均利用率受输入尺寸与 batch=2 限制。
- FCOS head: 亲和中等。四层小卷积塔 + GroupNorm + 分类/回归/中心度/offset 分支均可执行, 但多尺度小 tensor 的 head 开销明显。
- CondInst dynamic mask: 亲和中等偏低。每图实例数动态变化, `MAX_PROPOSALS=450`; 动态参数 reshape、group conv、mask upsample、boundary loss 增加 Vector/MTE 和调度压力。
- NMS 与 evaluator: 亲和低。NMS 走 CPU, SOBA/SOAP 指标为 numpy/pycocotools 路径, 对端到端吞吐影响明显。
- 分布式训练: 原生 Detectron2 launcher 没有 NPU local rank 绑定和 NPU DDP backend 选择, 2 进程入口失败。多卡数据并行需要替换 launcher、初始化 NPU 支持的通信后端, 并在每个 rank 显式绑定 `torch.npu.set_device(local_rank)`。

**profiler 摘要**

没有生成 torch_npu profiler trace。亲和性判断依据为完整评测输出、训练 metrics、`npu-smi` 运行期采样、报错栈和代码路径审计。

## 5. 阻塞项
| 阻塞点 | 原因 | 是否硬阻塞 | CANN/AscendC 替代方案 | 兜底 |
|---|---|---|---|---|
| ModulatedDeformConv 无原生 NPU 实现 | 原实现依赖 Detectron2/AdelaiDet C++/CUDA 扩展 | 对 CUDA 路径逐算子等价是硬阻塞; 对跑通和指标复核不是硬阻塞 | 基于 AscendC/CANN 实现 DCNv2 / deformable conv, 或接入官方可用 NPU deformable conv | 普通 conv2d 兼容实现, README 指标接近 |
| `replication_pad2d_backward` 失败 | NPU PadV3GradAiCore 动态 kernel 配置失败 | 对训练是硬阻塞 | 使用支持该 backward 的 CANN/torch_npu 版本或写自定义 pad backward | slice + expand + cat 等价重写, 已通过训练 |
| NMS 回 CPU | detectron2 `_C` 没有构建, batched NMS 无 NPU 扩展路径 | 否 | 使用 NPU NMS 算子或实现 AscendC NMS | 单图 CPU NMS |
| 2 进程 DDP 失败 | Detectron2 launcher 使用 CUDA 设备假设; NPU 设备没有绑定到 local rank, DDP backend 不匹配 | 对单卡不是硬阻塞; 对多卡训练是硬阻塞 | NPU-aware launch + HCCL/torch_npu 分布式后端 + rank/device 显式绑定 | 单卡训练 |
| SOBA PythonAPI Cython `_mask` 没有构建 | 原 Cython 扩展依赖本地编译链 | 否 | 编译 SOBA Cython 扩展 | 复用 `pycocotools._mask` |

## 6. 结论
- 运行方案: NPU+CPU 混合部署。单卡训练、单卡推理和两组 README 指标均完成; NMS 与 SOBA/SOAP 评估留在 CPU。
- 训练结论: 项目包含训练, 且 NPU 上已完成 200 iter 前后向、loss、optimizer step、checkpoint 写出。训练关键修正是 `aligned_bilinear` replicate pad 的等价改写; 多卡训练需要 NPU 分布式 launcher/backend 适配。
- 精度结论: SOBA-testing 与 SOBA-challenge 的 box AP 基本对齐 README; mask/SOAP 指标约 1 个点下降, 与 deformable conv 兼容替代路径一致。
- 亲和性结论: 主干卷积和 FCOS head 对 950PR 亲和; 动态 mask、replicate pad 等价重写、CPU NMS、Python 数据/评估链路拉低端到端利用率。若目标是进一步提速和逼近 CUDA 算子语义, 优先级为 NPU deformable conv、NPU NMS、NPU-aware DDP、数据管线并行化。
