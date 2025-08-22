安装环境：
```shell
# 安装opencompass依赖
pip install evalscope[opencompass] -U
```
数据准备：
```python
from evalscope.backend.opencompass import OpenCompassBackendManager
# 显示支持的数据集名称列表
OpenCompassBackendManager.list_datasets()
```
支持从 ModelScope 自动下载数据集，要启用此功能，请设置环境变量：
```shell
export DATASET_SOURCE=ModelScope
```

Qwen3-4B-SFT

+-----------+-----------------+-----------------+---------------+-------+
| task_type |     metric      |  dataset_name   | average_score | count |
+-----------+-----------------+-----------------+---------------+-------+
|   exam    | AverageAccuracy |      ceval      |     0.665     |  200  |
|   math    |  AveragePass@1  |    math_500     |     0.665     |  200  |
|   exam    | AverageAccuracy |    mmlu_pro     |     0.395     |  200  |
| knowledge |  AveragePass@1  |      gpqa       |    0.2323     |  198  |
|   code    |     Pass@1      | live_code_bench |    0.2363     |  182  |
+-----------+-----------------+-----------------+---------------+-------+

Qwen3-4B-SFT-QAT-w8a8

2025-08-19 10:02:38,516 - evalscope - INFO - dataset_level Report:
+-----------+-----------------+-----------------+---------------+-------+
| task_type |     metric      |  dataset_name   | average_score | count |
+-----------+-----------------+-----------------+---------------+-------+
|   exam    | AverageAccuracy |      ceval      |     0.68      |  200  |
|   math    |  AveragePass@1  |    math_500     |     0.71      |  200  |
|   exam    | AverageAccuracy |    mmlu_pro     |     0.425     |  200  |
| knowledge |  AveragePass@1  |      gpqa       |    0.2879     |  198  |
|   code    |     Pass@1      | live_code_bench |    0.2198     |  182  |
+-----------+-----------------+-----------------+---------------+-------+

Qwen3-4B

+-----------+-----------------+-----------------+---------------+-------+
| task_type |     metric      |  dataset_name   | average_score | count |
+-----------+-----------------+-----------------+---------------+-------+
|   exam    | AverageAccuracy |      ceval      |     0.855     |  200  |
|   math    |  AveragePass@1  |    math_500     |     0.935     |  200  |
|   exam    | AverageAccuracy |    mmlu_pro     |     0.715     |  200  |
| knowledge |  AveragePass@1  |      gpqa       |    0.5606     |  198  |
|   code    |     Pass@1      | live_code_bench |    0.4396     |  182  |
+-----------+-----------------+-----------------+---------------+-------+

Qwen3-4B-w8a8
+-----------+-----------------+-----------------+---------------+-------+
| task_type |     metric      |  dataset_name   | average_score | count |
+-----------+-----------------+-----------------+---------------+-------+
|   exam    | AverageAccuracy |      ceval      |     0.81      |  200  |
|   math    |  AveragePass@1  |    math_500     |     0.93      |  200  |
|   exam    | AverageAccuracy |    mmlu_pro     |     0.685     |  200  |
| knowledge |  AveragePass@1  |      gpqa       |    0.5253     |  198  |
|   code    |     Pass@1      | live_code_bench |    0.4615     |  182  |
+-----------+-----------------+-----------------+---------------+-------+

Qwen3-4B-gen-SFT

+-----------+-----------------+-----------------+---------------+-------+
| task_type |     metric      |  dataset_name   | average_score | count |
+-----------+-----------------+-----------------+---------------+-------+
|   exam    | AverageAccuracy |      ceval      |     0.755     |  200  |
|   math    |  AveragePass@1  |    math_500     |     0.915     |  200  |
|   exam    | AverageAccuracy |    mmlu_pro     |     0.61      |  200  |
| knowledge |  AveragePass@1  |      gpqa       |    0.4848     |  198  |
|   code    |     Pass@1      | live_code_bench |    0.3571     |  182  |
+-----------+-----------------+-----------------+---------------+-------+

Qwen3-4B-gen-SFT-w8a8


Qwen3-4B-gen-SFT-QAT-w8a8

+-----------+-----------------+-----------------+---------------+-------+
| task_type |     metric      |  dataset_name   | average_score | count |
+-----------+-----------------+-----------------+---------------+-------+
|   exam    | AverageAccuracy |      ceval      |     0.695     |  200  |
|   math    |  AveragePass@1  |    math_500     |     0.715     |  200  |
|   exam    | AverageAccuracy |    mmlu_pro     |     0.515     |  200  |
| knowledge |  AveragePass@1  |      gpqa       |    0.3131     |  198  |
|   code    |     Pass@1      | live_code_bench |    0.2198     |  182  |
+-----------+-----------------+-----------------+---------------+-------+
