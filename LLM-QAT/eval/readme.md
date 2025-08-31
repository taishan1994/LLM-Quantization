安装环境：
```shell
# 安装opencompass依赖
pip install evalscope[opencompass]==0.17.1
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
