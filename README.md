# 2024 年自然语言处理训练与应用

## 安装依赖

本项目建议使用python3.10版本的python，如果电脑没有安装CUDA，则使用下面命令安装CPU版本的依赖：
```shell
pip install torch transformers datasets numpy
``` 

如果电脑已经有CUDA，则安装对应CUDA版本的torch，其他模块正常安装就可以
```shell
pip3 install torch --index-url https://download.pytorch.org/whl/cu121
```

## 不同模块作用

- Bert.py: 对情感数据集进行分类，`train()`为训练，`test()`为测试，第一次使用需要先训练，然后再进行测试。