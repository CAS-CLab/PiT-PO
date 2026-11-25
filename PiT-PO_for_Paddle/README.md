这是 [PiT-PO](https://github.com/CAS-CLab/PiT-PO) 的 Paddle 版本。

环境：
Python 3.10、PaddlePaddle GPU 3.3.1、PaddleFormers 0.4.1。

```bash
source scripts/runtime_env.sh
python -m pip install paddlepaddle-gpu==3.3.1
python -m pip install -r requirements.txt
```

模型接口调用方式：
通过 `--model_path` 指定本地模型目录或 PaddleFormers 模型名称。使用 `AutoTokenizer.from_pretrained()` 和 `AutoModelForCausalLM.from_pretrained()` 加载模型，通过 `model.generate()` 生成候选方程，使用 LoRA GRPO 更新模型参数。

运行方式：
入口为 [launch_grpo.py](launch_grpo.py)，以下命令使用 GPU 0 进行 oscillator1 的 2500 个候选方程搜索：

```bash
source scripts/runtime_env.sh
python launch_grpo.py \
  --problem oscillator1 \
  --model_path /finder/wangboxiao/LLM-Research/Llama-3.2-3B-Instruct \
  --gpu 0 --dtype bfloat16 --max_samples 2500 \
  --output_dir artifacts/oscillator1
```

`--problem` 支持 `bactgrow`、`oscillator1`、`oscillator2` 和 `stressstrain`。运行目录保存 `training.log`、`config.json`、`run_summary.json`、`reward_curves/` 训练曲线、`checkpoints/` 定期适配器及 `final_adapter/` 最终适配器。

测试日志位于 [train_log](train_log/)。
包含 `bactgrow`、`oscillator1`、`oscillator2` 和 `stressstrain` 四个问题的 Llama-3.2-3B-Instruct 实验日志，包括训练输出、指标、运行汇总和训练曲线。
