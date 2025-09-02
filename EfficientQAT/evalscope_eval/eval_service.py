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

model_name='qwen3-4B-awq'
port = "18003"

task_cfg = TaskConfig(
    model=model_name,
    api_url='http://192.168.16.6:{}/v1/chat/completions'.format(port),
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
    use_cache="outputs/{}/".format(model_name)
)

run_task(task_cfg=task_cfg)