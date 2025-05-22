# Tactile Paving Analysis (盲道分析)

## 项目简介

本项目旨在通过图像分析技术，利用 OpenAI GPT-4o 模型评估城市环境中盲道（触觉铺装）的合规性与安全性。脚本会根据一系列预设问题对输入的盲道图片进行分析，并生成结构化的评估结果。

## 功能特性

-   **图像分析**：使用 GPT-4o 对盲道图片进行多维度评估。
-   **批量处理**：支持处理指定文件夹内的所有图片。
-   **并发处理**：利用 `asyncio` 实现并发 API 请求，提高处理效率。
-   **结构化输出**：为每张图片生成 JSON 格式的原始 API 响应和 CSV 格式的判断结果。
-   **可配置问题列表**：评估问题可灵活配置和扩展。
-   **环境变量管理 API 密钥**：安全地管理 OpenAI API 密钥。

## 技术栈

-   Python 3
-   OpenAI API (gpt-4o)
-   Pandas
-   Asyncio
-   Httpx
-   Tqdm

## 安装与设置

1.  **克隆仓库**:
    ```bash
    git clone https://github.com/ChenLin-99/TactilePavingAnalysis.git
    cd TactilePavingAnalysis
    ```

2.  **创建并激活虚拟环境** (推荐):
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # macOS/Linux
    # venv\Scripts\activate    # Windows
    ```

3.  **安装依赖**:
    ```bash
    pip install -r requirements.txt 
    ```

4.  **设置 OpenAI API 密钥**:
    本项目需要 OpenAI API 密钥才能运行。请将您的 API 密钥设置为名为 `OPENAI_API_KEY` 的环境变量。

    在 macOS/Linux 系统中，可以在终端执行:
    ```bash
    export OPENAI_API_KEY="sk-YOUR_ACTUAL_API_KEY"
    ```
    (请将 `sk-YOUR_ACTUAL_API_KEY` 替换为您真实的 OpenAI API 密钥)。为了永久生效，可以将此行添加到您的 shell 配置文件中 (如 `~/.zshrc` 或 `~/.bash_profile`)，然后执行 `source ~/.zshrc` (或相应文件)。

    在 Windows 系统中，可以在命令提示符或 PowerShell 中设置环境变量。

    **重要提示：没有正确设置 `OPENAI_API_KEY` 环境变量，脚本将无法运行。**

## 使用方法

主要通过运行 `0522_chinese.py` 脚本来进行分析。

```bash
python3 0522_chinese.py
```

-   脚本默认以批处理模式运行，处理 `data/` 文件夹中的所有图片。
-   结果将保存在 `results_batch/` 文件夹中。每张图片会生成一个 `_raw_response.json` 文件和一个 `_judgments.csv` 文件。
-   可以通过修改脚本顶部的 `TEST_MODE = True` 来启用测试模式，该模式仅处理 `TEST_MODE_IMAGE_PATH` 指定的单张图片，并将结果输出到 `results/` 文件夹。

## `0522_chinese.py` 脚本逻辑详解

`0522_chinese.py` 是本项目的核心脚本，其主要工作流程如下：

1.  **全局配置加载**：
    *   加载 OpenAI API 密钥 (从 `OPENAI_API_KEY` 环境变量)。
    *   初始化 `AsyncOpenAI` 客户端。
    *   定义数据输入 (`DATA_FOLDER`)、批处理输出 (`BATCH_OUTPUT_FOLDER`)、测试模式图片路径 (`TEST_MODE_IMAGE_PATH`) 和测试模式输出 (`OUTPUT_FOLDER`) 等路径。
    *   配置测试模式开关 (`TEST_MODE`)、测试问题数量 (`NUM_TEST_QUESTIONS`) 和并发请求上限 (`CONCURRENT_REQUEST_LIMIT`)。

2.  **问题定义 (`QUESTIONS_DATA`)**：
    *   一个多行字符串，包含了所有用于评估盲道的中文问题，每个问题都有一个唯一的ID（如 "1.1.1"）。

3.  **核心函数**：
    *   `encode_image_to_base64(image_path)`: 将图片文件编码为 Base64 字符串，以便发送给 API。
    *   `format_questions(questions_as_string, num_questions_to_format=None)`: 将 `QUESTIONS_DATA` 格式化为 API 请求所需的文本块。在测试模式下，可以限制问题数量。
    *   `parse_questions_for_ids(questions_as_string)`: 从 `QUESTIONS_DATA` 中解析出问题ID和问题文本的元组列表。
    *   `analyze_image_with_gpt4o(image_path, questions_text_block, question_ids_to_evaluate, is_test_mode, semaphore)`:
        *   这是与 OpenAI API 交互的核心异步函数。
        *   使用 `semaphore` 控制并发请求数量。
        *   构建发送给 GPT-4o 模型的请求体，包含图片 (Base64编码) 和格式化后的问题文本。
        *   请求模型以 JSON 对象格式返回响应。
        *   包含错误处理机制 (API连接错误、速率限制错误等)。
    *   `extract_answers(json_string, question_ids_expected)`: 从 API 返回的 JSON 响应中提取每个问题的判断结果 (0 表示否，1 表示是，-1 表示错误或未回答)。
    *   `save_test_mode_results(df, raw_response_json, image_name)`: 在测试模式下保存结果到 `results/` 文件夹，包括 CSV 和原始 JSON。
    *   `save_batch_mode_results(df, raw_response_json, image_name)`: 在批处理模式下保存结果到 `results_batch/` 文件夹。

4.  **运行模式**：
    *   `run_test_mode()`: 测试模式执行流程。处理单张图片，输出详细日志和结果。
    *   `process_batch()`: 批处理模式执行流程。
        *   遍历 `DATA_FOLDER` 中的所有图片。
        *   使用 `asyncio.gather` 和 `tqdm_asyncio` 并发处理图片。
        *   为每张图片调用 `analyze_image_with_gpt4o`。
        *   提取答案并保存结果。

5.  **主执行逻辑 (`if __name__ == "__main__":`)**：
    *   根据 `TEST_MODE` 的值选择执行 `run_test_mode()` 或 `process_batch()`。

**API 密钥提醒**：再次强调，请务必在运行脚本前正确设置 `OPENAI_API_KEY` 环境变量。如果未设置，脚本会在启动时抛出 `ValueError`。

## 目录结构

```
TactilePavingAnalysis/
├── .gitignore          # 指定 Git 忽略的文件和目录
├── 0522.py             # 原始英文版脚本 (参考用)
├── 0522_chinese.py     # 主要的中文分析脚本
├── data/               # 存放待分析的图片
│   ├── image1.jpg
│   └── ...
├── results/            # 测试模式下的输出结果
├── results_batch/      # 批处理模式下的输出结果
├── venv/               # Python 虚拟环境 (被 .gitignore 忽略)
└── README.md           # 本文件
```
