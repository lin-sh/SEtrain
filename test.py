import torch

# 检查是否有可用的 GPU
if torch.cuda.is_available():
    # 获取 GPU 数量
    num_gpus = torch.cuda.device_count()

    print(f"Number of available GPUs: {num_gpus}")

    # 打印每个 GPU 的名称
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("No GPU available.")
