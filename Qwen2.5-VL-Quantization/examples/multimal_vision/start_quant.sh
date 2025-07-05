cd llm-compressor
pip install -e .

pip install transformers==4.51.3

cd llm-compressor/examples/multimodal_vision
cp quant_scheme.py /usr/local/lib/python3.10/dist-packages/compressed_tensors/quantization/quant_scheme.py

export CUDA_VISIBLE_DEVICES=7
python3 qwen_2_5_vl_example.py
