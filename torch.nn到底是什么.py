from pathlib import Path
import requests
import torch
# DATA_PATH = Path("data")
# # print(DATA_PATH)
# PATH = DATA_PATH / "mnist"
# print(PATH)
# PATH.mkdir(parents=True, exist_ok=True)
# #
# URL = "http://deeplearning.net/data/mnist/"
# FILENAME = "mnist.pkl.gz"
#
# if not (PATH / FILENAME).exists():
#     print("开始下载数据")
#     content = requests.get(URL + FILENAME).content
#     (PATH / FILENAME).open("wb").write(content)
#     print("结束下载数据")
#
print(torch.randn(1,1,3))