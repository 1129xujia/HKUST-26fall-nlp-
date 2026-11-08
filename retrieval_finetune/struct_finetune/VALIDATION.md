# 代码迁移验证（2026-11-08）

本次只迁移训练与评测代码，没有重新执行正式 2000 步训练或完整四组最终测试。

在仓库根目录，使用原项目 `.venv_acceptance` 环境完成：

- `python -m pytest -q retrieval_finetune/struct_finetune/test_experiment.py`：3 passed。
- `bash -n retrieval_finetune/struct_finetune/run.sh`：通过。
- prepare/evaluate CLI 导入及 `--help`：通过。
- 使用原始 MultiDoc2Dial 运行 prepare：4110 个 struct passage；train20607/dev160/test400；三个输出文件 SHA256 与原 AutoDL 冻结输入完全一致。
- CPU 单线程真实训练：原始 BGE、全12层、2步、batch2；保存 checkpoint 后重载最大误差0；耗时6.28秒。仅验证训练链路，不代表正式训练质量或 GPU 性能。

冻结样本 SHA256：

| 划分 | SHA256 |
|---|---|
| train | ea63384bf2bd0214560530c374e9e7c278fff2b5cb74458efdff0d0c8775cac3 |
| dev | 0a102464b9ab1a9ee0f66dd1072f70508c0082fb366c38eb0c644d4bf5c7a25b |
| test | 88116ce0f82cd01e05951757448cb086acc7a85d3168b3a58e1c33121b849ad6 |

测试输出写入系统临时目录，不放入代码仓库。新的机器按 README 准备环境、数据与本地模型后先运行 smoke；本次未在新的 AutoDL 镜像重新验证 CUDA 环境。

## 本地完整入口复验

补齐 Git 忽略的冻结数据、struct 语料及基础模型后，执行：

```bash
PYTHON=/Users/leo/Desktop/hkust/5018/project/.venv_acceptance/bin/python DEVICE=cpu /bin/bash retrieval_finetune/struct_finetune/run.sh smoke
```

结果：3项测试通过，两步 CPU 全参数训练完成，`smoke_gpu/checkpoint-2/` 保存并重载成功。macOS 自动使用系统 Bash；smoke 默认单线程。上述数据、模型和 smoke 输出均被 Git 忽略。
