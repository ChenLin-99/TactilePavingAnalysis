import os
import base64
import pandas as pd
from openai import AsyncOpenAI, APIConnectionError, RateLimitError 
import httpx
import json
import asyncio 
from tqdm.asyncio import tqdm_asyncio 

# ========== 全局配置 ==========
API_KEY = os.environ.get("OPENAI_API_KEY") # 从环境变量读取API密钥
if not API_KEY:
    raise ValueError("请设置 OPENAI_API_KEY 环境变量。例如：export OPENAI_API_KEY='your_api_key_here'")

client = AsyncOpenAI(api_key=API_KEY) 
print("Initializing AsyncOpenAI client...")

# 路径配置
DATA_FOLDER = "/Users/chenlin99/Code/CHB/data" 
TEST_MODE_IMAGE_PATH = "/Users/chenlin99/Code/CHB/data/2c49fe04e349b28dc3839aa5450e0d63.jpg"
OUTPUT_FOLDER = "/Users/chenlin99/Code/CHB/results" 
BATCH_OUTPUT_FOLDER = "/Users/chenlin99/Code/CHB/results_batch" 

# 测试模式开关和配置
TEST_MODE = False # 控制是否启用测试模式
NUM_TEST_QUESTIONS = None # 测试模式下处理的问题数量，None表示处理所有问题
CONCURRENT_REQUEST_LIMIT = 50 # 批处理时并发API请求的上限

# 问题清单 (保持不变)
QUESTIONS_DATA = """
1.1.1 盲道砖之间是否存在明显的裂缝或过宽的缝隙？
1.1.2 盲道与相邻路面之间是否存在高度差？
1.1.3 盲道的不同部分之间是否存在较大的颜色差异？
1.1.4 盲道上的触觉点或条纹是否严重磨损或缺失？
1.1.5 导向型盲道是否直接引导至楼梯、栅栏或其他障碍物？
1.1.6 盲道的连续性是否被检查井盖或管道出口打断？
1.1.7 在需要设置盲道的位置（如地铁入口、公交车站）是否完全缺失盲道？
1.1.8 这段盲道是否包含两个以上超过90度的转弯？
1.2.1 是否有行人长时间停留在盲道上？
1.2.2 是否有快递员或外卖骑手临时停靠在盲道上或旁边？
1.2.3 是否有清洁车辆或卫生工具占据盲道？
1.2.4 盲道上是否有非机动车（如自行车、电动自行车）停放？
2.1.1 盲道是否距离墙壁不足250毫米？
2.1.2 盲道是否距离绿化带或种植区不足250毫米？
2.1.3 盲道是否距离树坑不足250毫米？
2.1.4 垃圾桶是否放置在盲道两侧250毫米范围内？
2.1.5 标志杆、灯柱或信息柱是否放置在盲道两侧250毫米范围内？
2.1.6 广告箱、邮箱或电气柜是否放置在盲道两侧250毫米范围内？
2.1.7 交通杆或导向柱是否位于盲道两侧250毫米范围内？
2.1.8 周围地砖颜色是否与盲道过于相似，导致难以区分？
2.1.9 周围地砖是否具有相似的触觉纹理，影响辨别？
2.1.10 盲道的边缘是否未明确定义，难以通过触摸检测？
2.2.1 盲道旁边是否有建筑障碍物，且没有设置临时引导砖？
2.2.2 是否有街头小贩在盲道250毫米范围内设摊？
2.2.3 是否有快递架或广告板临时放置在盲道附近？
2.2.4 是否有共享单车停放在盲道250毫米范围内？
2.2.5 是否有电动自行车、踏板车或其他非机动车阻塞盲道边缘？
3.1.1 当前人行道宽度是否小于2.0米？
3.1.2 人行道上是否有障碍物使有效通行宽度小于2.0米？
3.1.3 路缘石是否完全缺失？
3.1.4 路缘石是否过高（大于15厘米），导致通行困难？
3.1.5 路缘石是否过低（小于3厘米），增加视障行人进入机动车道的风险？
3.1.6 人行道与机动车道之间是否没有明确分隔？
3.1.7 人行道与道路是否处于同一水平，没有高度差？
3.2.1 建筑材料是否堆放在盲道或人行道上？
3.2.2 是否有堆积的绿化垃圾（如落叶、树枝）？
3.2.3 装修垃圾或废弃家具是否堆放在盲道附近？
3.2.4 临时建筑工地是否没有设置绕行通道或警示标志？
3.2.5 路边摊贩是否占据人行道/盲道空间且没有设置绕行引导？
3.2.6 商店是否将商品或家具摆放在盲道或人行道区域？
"""

def parse_questions_for_ids(questions_as_string):
    questions = []
    for line in questions_as_string.strip().split('\n'):
        if line.strip():
            parts = line.strip().split(" ", 1)
            if len(parts) == 2:
                questions.append((parts[0], parts[1]))
    return questions

def encode_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"错误: 图片文件 {image_path} 未找到。")
        return None
    except Exception as e:
        print(f"读取或编码图片 {image_path} 时出错: {e}")
        return None

def format_questions(questions_as_string, num_questions_to_format=None):
    formatted_questions = []
    questions_processed_count = 0
    for line in questions_as_string.strip().split('\n'):
        if line.strip():
            if num_questions_to_format is not None and questions_processed_count >= num_questions_to_format:
                break
            formatted_questions.append(line.strip())
            questions_processed_count += 1
    return "\n".join(formatted_questions)

async def analyze_image_with_gpt4o(image_path, questions_text_block, question_ids_to_evaluate, is_test_mode, semaphore):
    base64_image = encode_image(image_path)
    if not base64_image:
        return None, image_path

    if not is_test_mode:
        pass 
    else:
        print(f"分析图片: {os.path.basename(image_path)}")
        try:
            image_size_kb = os.path.getsize(image_path) / 1024
            print(f"图片大小: {image_size_kb:.2f} KB")
        except OSError:
            pass

    system_prompt_content = """
你是一个专业的图像分析助手，专门评估图片中的盲道是否存在问题。我将为你提供一张图片和一份问题清单。
请仔细分析图片，并根据问题清单逐项进行评估。

你的任务是生成一个JSON对象，该对象包含一个名为 "evaluations" 的数组。
数组中的每个元素都是一个JSON对象，代表对一个问题的评估，包含以下字段：
- "question_id": 问题的唯一标识符 (例如 "1.1.1")。
- "question_text": 问题的原文。
- "analysis": 你对图片中与该问题相关部分的详细分析和观察结果。
- "judgment": 你的最终判断。如果问题描述的情况存在，请返回 1；如果不存在，请返回 0。如果无法判断，请返回 -1。

重要提示：
1.  请确保你的回答严格按照指定的JSON格式。
2.  对于每个问题，请务必同时提供 "analysis" 和 "judgment"。
3.  在 "analysis" 中清晰描述你的判断依据。
4.  问题文本应该只在 "question_text" 字段中出现一次，不要在 "analysis" 或 "judgment" 中重复问题。
"""
    user_prompt_text = f"""
以下是需要评估的问题列表：
{questions_text_block}

请根据上述图片和问题列表，生成JSON格式的评估报告。"""

    full_response_content = ""
    async with semaphore: 
        try:
            if is_test_mode:
                print("开始调用 OpenAI API (JSON模式, 流式获取完整响应)...")
            
            start_time = pd.Timestamp.now()
            stream = await client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt_content},
                    {"role": "user", "content": [
                        {"type": "text", "text": user_prompt_text},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]}
                ],
                max_tokens=8192,
                temperature=0.2,
                stream=True,
                response_format={"type": "json_object"}
            )

            if is_test_mode:
                print("GPT响应 (流式接收JSON):")
                print("==================================================")
            
            async for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content_piece = chunk.choices[0].delta.content
                    if is_test_mode:
                        print(content_piece, end='', flush=True)
                    full_response_content += content_piece
            
            if is_test_mode:
                print("\n==================================================")
                end_time = pd.Timestamp.now()
                print(f"API调用完成，耗时: {(end_time - start_time).total_seconds():.2f} 秒")

        except APIConnectionError as e:
            print(f"API连接错误 (图片: {os.path.basename(image_path)}): {e}")
            return None, image_path
        except RateLimitError as e:
            print(f"API速率限制错误 (图片: {os.path.basename(image_path)}): {e}. 等待60秒后重试...")
            await asyncio.sleep(60)
            return await analyze_image_with_gpt4o(image_path, questions_text_block, question_ids_to_evaluate, is_test_mode, semaphore) 
        except Exception as e:
            print(f"调用OpenAI API时发生错误 (图片: {os.path.basename(image_path)}): {e}")
            return None, image_path

    return full_response_content, image_path

def extract_answers(json_string, question_ids_expected):
    if not json_string:
        print("⚠️ API响应为空，无法提取答案。")
        return {}
    try:
        data = json.loads(json_string)
        if "evaluations" not in data or not isinstance(data["evaluations"], list):
            print("⚠️ API响应中未找到 'evaluations' 数组或格式不正确。")
            return {}

        answers = {}
        for evaluation in data["evaluations"]:
            q_id = evaluation.get("question_id")
            judgment = evaluation.get("judgment")
            if q_id and isinstance(judgment, int):
                answers[q_id] = judgment
            else:
                print(f"警告: 问题 {q_id} 的评估结果格式不正确或judgment缺失。")
        
        final_answers = {qid: answers.get(qid, -1) for qid in question_ids_expected}
        return final_answers

    except json.JSONDecodeError:
        print(f"⚠️无法解析API返回的JSON: {json_string[:200]}...") 
        return {qid: -1 for qid in question_ids_expected} 
    except Exception as e:
        print(f"提取答案时发生未知错误: {e}")
        return {qid: -1 for qid in question_ids_expected}

def save_test_mode_results(df, raw_response_content, image_name):
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    base_filename = os.path.splitext(image_name)[0]
    excel_path = os.path.join(OUTPUT_FOLDER, f"{base_filename}_evaluation_gpt4o.xlsx")
    csv_path = os.path.join(OUTPUT_FOLDER, f"{base_filename}_evaluation_gpt4o.csv")
    raw_output_path = os.path.join(OUTPUT_FOLDER, f"{base_filename}_gpt_raw_output.txt")

    try:
        df.to_excel(excel_path, index=False)
        print(f"✅ Excel已保存至: {excel_path}")
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"✅ CSV已保存至: {csv_path}")
    except Exception as e:
        print(f"保存Excel/CSV文件时出错: {e}")

    if raw_response_content:
        try:
            with open(raw_output_path, 'w', encoding='utf-8') as f:
                f.write(raw_response_content)
            print(f"原始API响应已保存至: {raw_output_path}")
        except Exception as e:
            print(f"保存原始API响应时出错: {e}")

async def process_single_image_test_mode():
    print(f"🔍 正在处理测试图片: {TEST_MODE_IMAGE_PATH}")
    questions_to_format = NUM_TEST_QUESTIONS if TEST_MODE else None
    questions_text_block = format_questions(QUESTIONS_DATA, questions_to_format)
    
    all_question_tuples = parse_questions_for_ids(QUESTIONS_DATA)
    question_ids_to_evaluate = [q_id for q_id, _ in all_question_tuples]
    if TEST_MODE and NUM_TEST_QUESTIONS is not None:
        question_ids_to_evaluate = question_ids_to_evaluate[:NUM_TEST_QUESTIONS]

    semaphore = asyncio.Semaphore(1) 
    raw_response, _ = await analyze_image_with_gpt4o(TEST_MODE_IMAGE_PATH, questions_text_block, question_ids_to_evaluate, is_test_mode=True, semaphore=semaphore)

    answers_map = extract_answers(raw_response, question_ids_to_evaluate)

    if not answers_map or all(v == -1 for v in answers_map.values()):
        print("⚠️ 未能从API响应中提取到有效答案或所有答案均无效。")
        data_for_df = {qid: [-1] for qid in question_ids_to_evaluate}
        problems_found_count = 0
    else:
        data_for_df = {qid: [answers_map.get(qid, -1)] for qid in question_ids_to_evaluate}
        problems_found_count = sum(1 for qid in question_ids_to_evaluate if answers_map.get(qid) == 1)

    main_df = pd.DataFrame(data_for_df)
    main_df.insert(0, "Image", os.path.basename(TEST_MODE_IMAGE_PATH))
    main_df["Problems_Found"] = problems_found_count
    
    save_test_mode_results(main_df, raw_response, os.path.basename(TEST_MODE_IMAGE_PATH))

    print("\n数据概览:")
    print(main_df.to_string())
    print(f"\n发现的问题数量: {problems_found_count}/{len(question_ids_to_evaluate)}")

async def process_batch():
    os.makedirs(BATCH_OUTPUT_FOLDER, exist_ok=True)
    print(f"🚀 批处理模式已启用，将处理 '{DATA_FOLDER}'中的所有图片。")
    print(f"📂 结果将保存到: {BATCH_OUTPUT_FOLDER}")

    image_files = [f for f in os.listdir(DATA_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    if not image_files:
        print(f"在 '{DATA_FOLDER}' 中未找到图片文件。")
        return

    print(f"发现 {len(image_files)} 张图片待处理。")

    questions_text_block = format_questions(QUESTIONS_DATA) 
    all_question_tuples = parse_questions_for_ids(QUESTIONS_DATA)
    question_ids_expected = [q_id for q_id, _ in all_question_tuples]

    semaphore = asyncio.Semaphore(CONCURRENT_REQUEST_LIMIT)

    async def process_image_task(image_filename):
        image_path = os.path.join(DATA_FOLDER, image_filename)
        raw_response, _ = await analyze_image_with_gpt4o(image_path, questions_text_block, question_ids_expected, is_test_mode=False, semaphore=semaphore)

        if raw_response:
            raw_output_filename = f"{os.path.splitext(image_filename)[0]}_raw_response.json"
            raw_output_path = os.path.join(BATCH_OUTPUT_FOLDER, raw_output_filename)
            try:
                parsed_json = json.loads(raw_response)
                with open(raw_output_path, 'w', encoding='utf-8') as f:
                    json.dump(parsed_json, f, ensure_ascii=False, indent=2)
            except json.JSONDecodeError:
                with open(raw_output_path, 'w', encoding='utf-8') as f:
                    f.write(raw_response)
                # print(f"提示: {image_filename} 的API响应不是有效JSON，已按原样保存为文本。") 
            except Exception as e:
                print(f"保存 {image_filename} 的原始响应时出错: {e}")

            answers_map = extract_answers(raw_response, question_ids_expected)
            
            if answers_map and not all(v == -1 for v in answers_map.values()):
                judgments_data = [answers_map.get(q_id, -1) for q_id in question_ids_expected]
                problems_found_count = sum(1 for j in judgments_data if j == 1)
                
                df_single_image = pd.DataFrame([judgments_data], columns=question_ids_expected)
                df_single_image.insert(0, "Image", image_filename)
                df_single_image["Problems_Found"] = problems_found_count

                csv_filename = f"{os.path.splitext(image_filename)[0]}_judgments.csv"
                csv_output_path = os.path.join(BATCH_OUTPUT_FOLDER, csv_filename)
                try:
                    df_single_image.to_csv(csv_output_path, index=False, encoding='utf-8-sig')
                except Exception as e:
                    print(f"保存 {image_filename} 的判断结果CSV时出错: {e}")
                return f"{image_filename}: ✔️ ({problems_found_count} 问题)"
            else:
                return f"{image_filename}: ⚠️ (提取答案失败)"
        else:
            return f"{image_filename}: ❌ (API调用失败)"

    tasks = [process_image_task(img_file) for img_file in image_files]
    
    for future in tqdm_asyncio.as_completed(tasks, desc="处理图片中", total=len(image_files)):
        try:
            result_message = await future
            # tqdm updates progress. Optionally print per-image result if not too verbose.
            # print(result_message) 
        except Exception as e:
            print(f"处理某图片时发生意外的异步错误: {e}")

    print("\n🎉 所有图片处理完成。")
    print(f"📂 请检查输出文件夹: {BATCH_OUTPUT_FOLDER}")

if __name__ == "__main__":
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(BATCH_OUTPUT_FOLDER, exist_ok=True)

    if TEST_MODE:
        print(f"🧪 测试模式已启用，将处理单张图片: {TEST_MODE_IMAGE_PATH}")
        if NUM_TEST_QUESTIONS:
             print(f"   仅处理前 {NUM_TEST_QUESTIONS} 个问题。")
        asyncio.run(process_single_image_test_mode())
    else:
        asyncio.run(process_batch())
