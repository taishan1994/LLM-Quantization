# LLM-QAT

使用Qwen3-4B进行QAT。

对原始的LLM-QAT进行了修改以支持Qwen3系列模型。

- 修改支持Qwen3的模型架构，主要是在models文件夹下进行重构，将相关线性层替换为量化的线性层。
- 使用trl中的SFTTrainer替换掉原始的sft trainer，以支持SFT数据格式的训练。
- 使用trl中的GKDTrainer替换掉原始的KDTrainer，以支持进行蒸馏学习的QAT。
- 使用llmc作为训练完成后的进一步RTN量化。
- 使用vllm作为部署框架部署量化前后的模型。
- 使用evalscope作为评测框架评测训练前后模型的结果。

# 环境依赖

```shell
torch==2.9.0.dev20250804+cu128
transformers==4.53.2
deepspeed==0.17.5
accelerate==1.9.0
numpy==2.2.6
jinja2==3.1.6
datasets==4.0.0
modelscope==1.28.1
pydantic==2.11.7
bitsandbytes==0.46.1
```

# QAT(量化感知训练)

下载预训练的权重，可以直接放在model_hub文件夹下，或者是任意的地方，到时候指定即可。

为了不破坏掉模型的原始分布，LLM-QAT使用的是模型自生成的数据，参考代码：`generate_sft_data.py`

首先需要参开service/vllm_service.sh下部署一个原始模型的vllm服务，然后根据generate_sft_data.py生成对应格式的数据。

这里使用的数据是：medical_o1_sft_Chinese.json，下载后可以放在data/medical_o1下面。使用的数据集路径需要再在train_qwen3_sft.py里面进行修改。

为了进一步分析数据对QAT的影响，可以直接使用生成后的数据进行SFT，使用脚本`run_train_qwen3_sft.sh`

```shell
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=15001 train_qwen3_sft.py \
--local_dir "model_hub/Qwen/Qwen3-4B" \
--input_model_filename "Qwen3/Qwen3-4B" \
--output_model_filename "qwen3-4B-gen-finetuned2" \
--do_train True \
--do_eval False \
--max_length 4096 \
--fp16 False \
--bf16 True \
--log_on_each_node False \
--logging_dir ./train_log \
--num_train_epochs 1 \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 1 \
--gradient_accumulation_steps 1 \
--save_strategy "steps" \
--save_steps 2000 \
--report_to "tensorboard" \
--save_total_limit 1 \
--learning_rate 2e-5 \
--weight_decay 0. \
--warmup_ratio 0. \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--tf32 False \
--gradient_checkpointing True \
--qat False 

```

- local_dir：晕训练模型的路径
- qat：是否使用QAT(量化感知训练)

其余参数默认即可。

使用`run_train_qwen3_sft_quant.sh`进行qat训练，与sft相比：

```shell
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=15001 train_qwen3_sft.py \
--local_dir "model_hub/Qwen/Qwen3-4B" \
--input_model_filename "Qwen3/Qwen3-4B" \
--output_model_filename "qwen3-4B-gen-quant-finetuned2" \
--do_train True \
--do_eval False \
--max_length 4096 \
--fp16 False \
--bf16 True \
--log_on_each_node False \
--logging_dir ./train_log \
--num_train_epochs 1 \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 1 \
--gradient_accumulation_steps 1 \
--save_strategy "steps" \
--save_steps 2000 \
--report_to "tensorboard" \
--save_total_limit 1 \
--learning_rate 2e-5 \
--weight_decay 0. \
--warmup_ratio 0. \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--tf32 False \
--gradient_checkpointing True \
--w_bits 8 \
--a_bits 8 \
--kv_bits 16 \
--qat True 
```

- qat：设置为True
- 设置w_bits、a_bits以及kv_bits，另外默认对权重是per-channel对称量化，对激活是per-token对称量化。

使用`run_train_qwen3_sft_quant_kd.sh`进行蒸馏学习的QAT训练，与qat相比：

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=15001 train_qwen3_sft.py \
--local_dir "model_hub/Qwen/Qwen3-4B" \
--input_model_filename "Qwen3/Qwen3-4B" \
--output_model_filename "qwen3-4B-gen-quant-finetuned2-kd" \
--do_train True \
--do_eval False \
--max_length 1024 \
--fp16 False \
--bf16 True \
--log_on_each_node False \
--logging_dir ./train_log \
--num_train_epochs 1 \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 1 \
--gradient_accumulation_steps 1 \
--save_strategy "steps" \
--save_steps 2000 \
--report_to "tensorboard" \
--save_total_limit 1 \
--learning_rate 2e-5 \
--weight_decay 0. \
--warmup_ratio 0. \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--tf32 False \
--gradient_checkpointing True \
--w_bits 8 \
--a_bits 8 \
--kv_bits 16 \
--qat True \
--use_kd True \
--lmbda 0.0 \
--beta 0.0 \
--seq_kd False
```

额外增加了四个参数：use_kd、lmbda、beta、seq_kd，具体是什么作用请参考trl的官方文档。

# Quantization(量化)

训练完成后保存的模型还是bf16格式的，需要进一步进行量化（注意训练时候的量化配置要和这里的对齐，比如训练使用w8a8，这里量化也要使用w8a8）。量化的框架为LightCompress，原名为llmc。首先是安装环境依赖，最后另建一个虚拟环境。

```shell
cd LightCompress
pip install -r requirements.txt
```

然后在configs/quantization/rtn_w_a.yml里面配置模型路径，最后运行：

```shell
bash scripts/run_llmc.sh 
```

里面默认使用的是configs/quantization/rtn_w_a.yml，可以酌情进行修改。

# Deploy(部署)

准备环境：`pip install vllm -U` 最好新建一个虚拟环境。

量化后的模型我们使用vllm进行部署，具体可以参考service下的文件，部署好后可以使用service/test_service.py进行测试服务是否正常。

# Eval(评估)

在部署好后就可以进行评估了，安装`pip install evalscope[opencompass]` 这个可以和vllm共用一个虚拟环境、

在eval/eval.py里面是评测脚本，这里简单说明下：

```python
import os
from evalscope import TaskConfig, run_task
from evalscope.collections import CollectionSchema, DatasetInfo, WeightedSampler, UniformSampler
from evalscope.utils.io_utils import dump_jsonl_data

schema = CollectionSchema(name='Qwen3', datasets=[
    CollectionSchema(name='English', datasets=[
        DatasetInfo(name='mmlu_pro', weight=1, task_type='exam', tags=['en'], args={'few_shot_num': 0}),
        #DatasetInfo(name='mmlu_redux', weight=1, task_type='exam', tags=['en'], args={'few_shot_num': 0}),
        #DatasetInfo(name='ifeval', weight=1, task_type='instruction', tags=['en'], args={'few_shot_num': 0}),
    ]),
    CollectionSchema(name='Chinese', datasets=[
        DatasetInfo(name='ceval', weight=1, task_type='exam', tags=['zh'], args={'few_shot_num': 0}),
        #DatasetInfo(name='iquiz', weight=1, task_type='exam', tags=['zh'], args={'few_shot_num': 0}),
    ]),
    CollectionSchema(name='Code', datasets=[
        DatasetInfo(name='live_code_bench', weight=1, task_type='code', tags=['en'], args={'few_shot_num': 0, 'subset_list': ['v5_v6'], 'extra_params': {'start_date': '2025-01-01', 'end_date': '2025-04-30'}}),
    ]),
    CollectionSchema(name='Math&Science', datasets=[
        DatasetInfo(name='math_500', weight=1, task_type='math', tags=['en'], args={'few_shot_num': 0}),
        #DatasetInfo(name='aime24', weight=1, task_type='math', tags=['en'], args={'few_shot_num': 0}),
        #DatasetInfo(name='aime25', weight=1, task_type='math', tags=['en'], args={'few_shot_num': 0}),
        DatasetInfo(name='gpqa', weight=1, task_type='knowledge', tags=['en'], args={'subset_list': ['gpqa_diamond'], 'few_shot_num': 0})
    ])
])

# get the mixed data
#mixed_data = WeightedSampler(schema).sample(100000000)  # set a large number to ensure all datasets are sampled
# dump the mixed data to a jsonl file
#dump_jsonl_data(mixed_data, 'outputs/qwen3_test.jsonl')

# 均匀采样，每个数据集采样200条，共1000条数据
data_path = 'outputs/uniform_mixed_data_1000.jsonl'
if not os.path.exists(data_path):
    sampler = UniformSampler(schema)
    mixed_data = sampler.sample(1000)
    dump_jsonl_data(mixed_data, data_path)

task_cfg = TaskConfig(
    model='Qwen3-4B-finetuned2',
    api_url='http://14.18.247.55:11777/v1/chat/completions',
    eval_type='service',
    datasets=[
        'data_collection',
    ],
    dataset_args={
        'data_collection': {
            'dataset_id': data_path,
            'filters': {'remove_until': '</think>'}  # 过滤掉思考的内容
        }
    },
    eval_batch_size=128,
    generation_config={
        'max_tokens': 30000,  # 最大生成token数，建议设置为较大值避免输出截断
        'temperature': 0.6,  # 采样温度 (qwen 报告推荐值)
        'top_p': 0.95,  # top-p采样 (qwen 报告推荐值)
        'top_k': 20,  # top-k采样 (qwen 报告推荐值)
        'n': 1,  # 每个请求产生的回复数量
    },
    timeout=6000000,  # 超时时间
    stream=True,  # 是否使用流式输出
    # limit=100,  # 设置为100条数据进行测试
    use_cache="outputs/20250817_194440"
)

run_task(task_cfg=task_cfg)
```

- 我们选择了mmlu_pro、ceval、live_code_bench、math_500、gpqa五个数据集，并且进行均匀采样，总共采样1000条数据。
- model：是我们启动vllm时设置的服务的模型的名称。
- api_url：启动服务后提供的url地址。

其余参数的说明见注释。最后直接运行这个文件就可以启动测试：`python3 eval.py`

# Result(结果)

首先是我们采样到的数据的分布：

| task_type |     metric      |  dataset_name   | count |
| :-------: | :-------------: | :-------------: | :---: |
|   exam    | AverageAccuracy |      ceval      |  200  |
|   math    |  AveragePass@1  |    math_500     |  200  |
|   exam    | AverageAccuracy |    mmlu_pro     |  200  |
| knowledge |  AveragePass@1  |      gpqa       |  198  |
|   code    |     Pass@1      | live_code_bench |  182  |

训练的结果：

| Model                     | ceval | math_500 | mmlu_pro | gpqa   | live_code_bench |
| ------------------------- | ----- | -------- | -------- | ------ | --------------- |
| Qwen3-4B                  | 0.855 | 0.935    | 0.715    | 0.5606 | 0.4396          |
| Qwen3-4B-w8a8             | 0.81  | 0.93     | 0.685    | 0.5253 | 0.4615          |
| Qwen3-4B-SFT              | 0.665 | 0.665    | 0.395    | 0.2323 | 0.2363          |
| Qwen3-4B-SFT-QAT-w8a8     | 0.68  | 0.71     | 0.425    | 0.2879 | 0.2198          |
| Qwen3-4B-gen-SFT          | 0.755 | 0.915    | 0.61     | 0.4848 | 0.3571          |
| Qwen3-4B-gen-SFT-w8a8     | 0.76  | 0.88     | 0.605    | 0.4394 | 0.3462          |
| Qwen3-4B-gen-SFT-QAT-w8a8 | 0.755 | 0.865    | 0.625    | 0.3939 | 0.3352          |
| Qwen3-4B-gen-KD-QAT-w8a8  | 0.83  | 0.92     | 0.69     | 0.5556 | 0.4505          |
| Qwen3-4B-Quarot-GPTQ      | 0.84  | 0.925    | 0.665    | 0.5606 | 0.4396          |

还是得结合知识蒸馏进行训练，最终QAT之后的结果在某些数据集上比直接RTN的效果要好。另外，与PTQ方法相比，各有优势吧。

# 补充

还可以进一步将torchao的量化实现集成到LLM-QAT里面，不过需要注意的是torchao暂时不支持激活per-token的对称量化。

