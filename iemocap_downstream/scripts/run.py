import os
import subprocess
import torch
print(torch.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# 假设这些变量已经被设置为所需的路径
manifest_path = r"C:\Users\ROG\Desktop\project\emotion2vec\iemocap_downstream\scripts\data"
model_path = r"C:\Users\ROG\Desktop\project\emotion2vec\upstream"
checkpoint_path = r"C:\Users\ROG\Desktop\project\emotion2vec\emotion2vec_base\emotion2vec_base.pt"
save_dir = r"C:\Users\ROG\Desktop\project\emotion2vec\iemocap_downstream\scripts\features_2"
# 假设您的模型有多层，您想要提取所有层的特征
layers = range(12)  # 举例，如果模型有12层

for layer in layers:
    true_layer = layer + 1
    print(f"Extracting features from layer {true_layer}")

    # 构建Python中的命令执行字符串
    command = [
        "python", "emotion2vec_speech_features.py",
        "--data", manifest_path,
        "--model", model_path,
        "--split", "train",
        "--checkpoint", checkpoint_path,
        "--save-dir", save_dir,
        "--layer", str(layer)
    ]

    # 使用subprocess运行命令
    subprocess.run(command)