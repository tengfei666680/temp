<!--
 * @Author: shsntf
 * @email: shantf@sugon.com
 * @Date: 2024-12-12 10:38:07
 * @LastEditTime: 2024-12-12 17:50:01
-->
# Baichuan

## 论文
- [https://arxiv.org/abs/2309.10305](https://arxiv.org/abs/2309.10305)

## 模型结构
Baichuan系列模型是由百川智能开发的开源大规模预训练模型，包含7B和13B等规模。其中，Baichuan-7B在大约1.2万亿tokens上训练的70亿参数模型，支持中英双语，上下文窗口长度为4096。Baichuan-13B是由百川智能继Baichuan-7B之后开发的包含130亿参数模型，它在高质量的语料上训练了1.4万亿tokens，超过LLaMA-13B 40%，是当前开源 13B 尺寸下训练数据量最多的模型。此外，百川智能还发布了对齐模型（Baichuan-13B-Chat），具有很强的对话能力。Baichuan 2 是百川智能推出的新一代开源大语言模型，采用 2.6 万亿Tokens 的高质量语料训练。
模型具体参数：
| 模型名称 | 隐含层维度 | 层数 | 头数 | 词表大小 | 位置编码 | 最大长 |
| -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| Baichuan-7B | 4,096 | 32 | 32 | 64,000 | RoPE | 4096 |
| Baichuan-13B | 5,120 | 40 | 	40 | 64,000 | ALiBi | 4096 |
| Baichuan 2-7B | 4,096 | 32 | 32 | 125,696 |  RoPE | 4096 |
| Baichuan 2-13B | 5,120 | 40 | 	40 | 125,696 |   ALiBi | 4096 |

<div align="center">
<img src="./docs/baichuan.jpg" width="400" height="300">
</div>

## 算法原理
Baichuan整体模型基于标准的Transformer结构，采用了和LLaMA一样的模型设计。其中，Baichuan-7B在结构上采用Rotary Embedding位置编码方案、SwiGLU激活函数、基于RMSNorm的Pre-Normalization。Baichuan-13B使用了ALiBi线性偏置技术，相对于Rotary Embedding计算量更小，对推理性能有显著提升。

<div align="center">
<img src="./docs/baichuan.png" width="450" height="300">
</div>

## 环境配置

### Docker（方法一）
提供[光源](https://www.sourcefind.cn/#/image/dcu/custom)拉取推理的docker镜像：

```
docker pull image.sourcefind.cn:5000/dcu/admin/base/pytorch:2.3.0-py3.10-dtk24.04.3-ubuntu20.04
# <Image ID>用上面拉取docker镜像的ID替换
# <Host Path>主机端路径
# <Container Path>容器映射路径
docker run -it --name baichuan_vllm --privileged --shm-size=128G  --device=/dev/kfd --device=/dev/dri/ --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --ulimit memlock=-1:-1 --ipc=host --network host --group-add video -v /opt/hyhal:/opt/hyhal:ro -v <Host Path>:<Container Path> <Image ID> /bin/bash
```
`Tips：若在K100/Z100L上使用，使用定制镜像docker pull image.sourcefind.cn:5000/dcu/admin/base/custom:vllm0.5.0-dtk24.04.1-ubuntu20.04-py310-zk-v1,K100/Z100L不支持awq量化`

### Dockerfile（方法二）
```
# <Host Path>主机端路径
# <Container Path>容器映射路径
docker build -t baichuan:latest .
docker run -it --name baichuan_vllm --privileged --shm-size=128G  --device=/dev/kfd --device=/dev/dri/ --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --ulimit memlock=-1:-1 --ipc=host --network host --group-add video -v /opt/hyhal:/opt/hyhal:ro -v <Host Path>:<Container Path> baichuan:latest /bin/bash
```

### Anaconda（方法三）
```
conda create -n baichuan_vllm python=3.10
```
关于本项目DCU显卡所需的特殊深度学习库可从[光合](https://developer.hpccube.com/tool/)开发者社区下载安装。
* DTK驱动：dtk24.04.3
* Pytorch: 2.3.0
* triton:2.1.0+das.opt1.dtk24042
* lmslim: 0.1.1+das.dtk24042
* xformers: 0.0.25+das.opt1.dtk24042
* vllm: 0.5.0+das.opt4.dtk24043
* python: python3.10

`Tips：需先安装相关依赖，最后安装vllm包`

## 数据集
无

## 推理

### 模型下载

**快速下载通道：**

| 基座模型 | chat模型 | GPTQ模型 |
| ------- | ------- | ------- |
| [Baichuan-7B](http://113.200.138.88:18080/aimodels/Baichuan-7B)   | [baichuan-7B-chat](https://hf-mirror.com/csdc-atl/baichuan-7B-chat)    | [baichuan-7B-GPTQ](https://hf-mirror.com/TheBloke/baichuan-7B-GPTQ)   |
| [Baichuan2-7B-Base](http://113.200.138.88:18080/aimodels/Baichuan2-7B-Base) | [Baichuan2-7B-Chat](http://113.200.138.88:18080/aimodels/Baichuan2-7B-Chat) | [Baichuan2-7B-Chat-GPTQ-Int4](https://hf-mirror.com/csdc-atl/Baichuan2-7B-Chat-GPTQ-Int4) |
| [Baichuan2-13B-Base](http://113.200.138.88:18080/aimodels/Baichuan2-13B-Base) |  | [Baichuan2-13B-Chat-GPTQ](https://hf-mirror.com/TheBloke/Baichuan2-13B-Chat-GPTQ) |

`Tips：若遇到报错：AttributeError:'BaiChuanTokenizer' object has no attribute 'sp_model',可以将模型文件'tokenization_baichuan.py'中BaichuanTokenizer类的super().__init__部分移动到__init__()部分的最后，如下图所示：`

<div align="center">
<img src="./docs/2jj09uf4.png" width="849.6" height="694.8">
</div>

`Tips：Tips：Baichuan2-13B-Chat-GPTQ（group-size=128）模型不支持vllm多卡推理，仅可以单卡推理`

### 离线批量推理

```bash
python examples/offline_inference.py
```
其中，`prompts`为提示词；`temperature`为控制采样随机性的值，值越小模型生成越确定，值变高模型生成更随机，0表示贪婪采样，默认为1；`max_tokens=16`为生成长度，默认为1；
`model`为模型路径；`tensor_parallel_size=1`为使用卡数，默认为1；`dtype="float16"`为推理数据类型，如果模型权重是bfloat16,需要修改为float16推理,`quantization="gptq"`为使用gptq量化进行推理,需下载以上GPTQ模型。`quantization="awq"`为使用awq量化进行推理,需下载以上AWQ模型。


### 离线批量推理性能测试
1、指定输入输出
```bash
python benchmarks/benchmark_throughput.py --num-prompts 1 --input-len 32 --output-len 128 --model baichuan-inc/Baichuan-7B -tp 1 --trust-remote-code --enforce-eager --dtype float16
```
其中`--num-prompts`是batch数，`--input-len`是输入seqlen，`--output-len`是输出token长度，`--model`为模型路径，`-tp`为使用卡数，`dtype="float16"`为推理数据类型，如果模型权重是bfloat16,需要修改为float16推理。若指定`--output-len  1`即为首字延迟。`-q gptq`为使用gptq量化模型进行推理。

2、使用数据集
下载数据集：
```bash
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

```bash
python benchmarks/benchmark_throughput.py --num-prompts 1 --model baichuan-inc/Baichuan-7B --dataset ShareGPT_V3_unfiltered_cleaned_split.json -tp 1 --trust-remote-code --enforce-eager --dtype float16
```
其中`--num-prompts`是batch数，`--model`为模型路径，`--dataset`为使用的数据集，`-tp`为使用卡数，`dtype="float16"`为推理数据类型，如果模型权重是bfloat16,需要修改为float16推理。`-q gptq`为使用gptq量化模型进行推理。


### api服务推理性能测试
1、启动服务端：
```bash
python -m vllm.entrypoints.openai.api_server  --model baichuan-inc/Baichuan-7B  --dtype float16 --enforce-eager -tp 1 
```

2、启动客户端：
```bash
python benchmarks/benchmark_serving.py --model baichuan-inc/Baichuan-7B --dataset ShareGPT_V3_unfiltered_cleaned_split.json  --num-prompts 1 --trust-remote-code
```
参数同使用数据集，离线批量推理性能测试，具体参考[benchmarks/benchmark_serving.py](benchmarks/benchmark_serving.py)


### OpenAI兼容服务
启动服务：
```bash
python -m vllm.entrypoints.openai.api_server --model baichuan-inc/Baichuan-7B --enforce-eager --dtype float16 --trust-remote-code
```
这里`--model`为加载模型路径，`--dtype`为数据类型：float16，默认情况使用tokenizer中的预定义聊天模板，`--chat-template`可以添加新模板覆盖默认模板,`-q gptq`为使用gptq量化模型进行推理。

列出模型型号：
```bash
curl http://localhost:8000/v1/models
```

### OpenAI Completions API和vllm结合使用
```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "baichuan-inc/Baichuan-7B",
        "prompt": "I believe the meaning of life is",
        "max_tokens": 7,
        "temperature": 0
    }'
```
或者使用[examples/openai_completion_client.py](examples/openai_completion_client.py)


### OpenAI Chat API和vllm结合使用
```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "baichuan-inc/Baichuan-7B",
        "messages": [
            {"role": "system", "content": "I believe the meaning of life is"},
            {"role": "user", "content": "I believe the meaning of life is"}
        ]
    }'
```
或者使用[examples/openai_chatcompletion_client.py](examples/openai_chatcompletion_client.py)
### **gradio和vllm结合使用**

1.安装gradio

```
pip install gradio
```

2.安装必要文件

    2.1 启动gradio服务，根据提示操作

```
python  gradio_openai_chatbot_webserver.py --model "baichuan-inc/Baichuan-7B" --model-url http://localhost:8000/v1 --temp 0.8 --stop-token-ids ""
```

    2.2 更改文件权限

打开提示下载文件目录，本项目中已经下载好所需文件frpc_linux_amd64_v0.3，输入以下命令给予权限

```
chmod +x frpc_linux_amd64_v0.*
```
   2.3 端口映射

```
ssh -L 8000:计算节点IP:8000 -L 8001:计算节点IP:8001 用户名@登录节点 -p 登录节点端口
```

3.启动OpenAI兼容服务

```
python -m vllm.entrypoints.openai.api_server --model baichuan-inc/Baichuan-7B --enforce-eager --dtype float16 --trust-remote-code --port 8000 --host "0.0.0.0"
```

4.启动gradio服务

```
python example/gradio_openai_chatbot_webserver.py --model "baichuan-inc/Baichuan-7B" --model-url http://localhost:8000/v1 --temp 0.8 --stop-token-ids --host "0.0.0.0" --port 8001"
```

5.使用对话服务

在浏览器中输入本地 URL，可以使用 Gradio 提供的对话服务。
## result
使用的加速卡:1张 DCU-K100_AI-64G
```
Prompt: 'I believe the meaning of life is', Generated text: ' to find purpose, happiness, and fulfillment. Here are some reasons why:\n\n1. Purpose: Having a sense of purpose gives life meaning and direction. It helps individuals set goals and work towards achieving them, which can lead to a sense of accomplishment and fulfillment.\n2. Happiness: Happiness is a fundamental aspect of life that brings joy and satisfaction.
```

### 精度
无

## 应用场景

### 算法类别
对话问答

### 热点应用行业
金融,科研,教育

## 源码仓库及问题反馈
* [https://developer.sourcefind.cn/codes/modelzoo/baichuan_vllm](https://developer.sourcefind.cn/codes/modelzoo/baichuan_vllm)

## 参考资料
* [https://github.com/vllm-project/vllm](https://github.com/vllm-project/vllm)
