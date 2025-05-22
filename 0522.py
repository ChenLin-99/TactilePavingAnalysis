import os
import re
import base64
import pandas as pd
import matplotlib.pyplot as plt
from openai import OpenAI

# ========== 初始化 OpenAI 客户端 ==========
API_KEY = os.environ.get("OPENAI_API_KEY") # Read API key from environment variable
if not API_KEY:
    raise ValueError("Please set the OPENAI_API_KEY environment variable. E.g., export OPENAI_API_KEY='your_api_key_here'")

client = OpenAI(api_key=API_KEY)

# ========== 文件路径配置 ==========
IMAGE_FOLDER = r"E:\处理\Other9_盲道_Chi\Data\地铁站 110\万寿寺"
OUTPUT_EXCEL = os.path.join(IMAGE_FOLDER, "blind_path_evaluation_gpt4o.xlsx")

# ========== 学术英文问卷 ==========
questions = [
    "1.1.1 Are there any significant cracks or excessively wide gaps between the tactile paving tiles?",
    "1.1.2 Is there any height difference between the tactile paving and the adjacent road surface?",
    "1.1.3 Are there large color differences between sections of the tactile paving?",
    "1.1.4 Are the tactile dots or bars on the paving severely worn or missing?",
    "1.1.5 Does the guidance paving lead directly to stairs, fences, or other obstacles?",
    "1.1.6 Is the continuity of the tactile paving interrupted by manholes or pipe outlets?",
    "1.1.7 Is tactile paving completely missing in locations where it is needed (e.g., metro entrances, bus stops)?",
    "1.1.8 Does this section of tactile paving contain more than two turns exceeding 90 degrees?",
    "1.2.1 Is there any pedestrian staying on the tactile paving for an extended period?",
    "1.2.2 Are there couriers or food delivery workers temporarily parked on or beside the tactile paving?",
    "1.2.3 Are cleaning vehicles or sanitation tools occupying the tactile paving?",
    "1.2.4 Are there non-motorized vehicles (e.g., bikes, e-bikes) parked on the tactile paving?",
    "2.1.1 Is the tactile paving within 250mm of a wall?",
    "2.1.2 Is the tactile paving within 250mm of greenbelt or planted area?",
    "2.1.3 Is the tactile paving within 250mm of a tree pit?",
    "2.1.4 Are garbage bins placed within 250mm on either side of the tactile paving?",
    "2.1.5 Are signposts, lampposts, or information columns placed within 250mm on either side?",
    "2.1.6 Are advertisement boxes, mailboxes, or electrical cabinets placed within 250mm on either side?",
    "2.1.7 Are traffic poles or guideposts located within 250mm on either side?",
    "2.1.8 Are surrounding floor tiles too similar in color to the tactile paving, making it hard to distinguish?",
    "2.1.9 Do surrounding floor tiles have similar tactile textures that impair differentiation?",
    "2.1.10 Are the edges of the tactile paving not clearly defined and difficult to detect by touch?",
    "2.2.1 Are there construction barriers directly adjacent to the tactile paving without temporary guiding tiles?",
    "2.2.2 Are there street vendors set up within 250mm of the tactile paving?",
    "2.2.3 Are delivery racks or advertisement boards temporarily placed near the tactile paving?",
    "2.2.4 Are shared bicycles parked within 250mm of the tactile paving?",
    "2.2.5 Are e-bikes, scooters, or other non-motorized vehicles blocking the tactile paving’s edge?",
    "3.1.1 Is the current sidewalk width less than 2.0 meters?",
    "3.1.2 Are there barriers on the sidewalk making the effective width less than 2.0 meters?",
    "3.1.3 Are curb stones completely missing?",
    "3.1.4 Is the curb too high (>15cm), creating difficulty in crossing?",
    "3.1.5 Is the curb too low (<3cm), increasing risk of blind pedestrians entering vehicular roads?",
    "3.1.6 Is there no clear separation between sidewalk and motor vehicle road?",
    "3.1.7 Is the sidewalk and the road on the same level, without height difference?",
    "3.2.1 Are building materials stored on the tactile or pedestrian paths?",
    "3.2.2 Is there accumulated green waste (e.g., fallen leaves, branches)?",
    "3.2.3 Are renovation wastes or abandoned furniture piled near the tactile paving?",
    "3.2.4 Are there temporary construction sites without bypass paths or warning signs?",
    "3.2.5 Are roadside vendors occupying pedestrian/tactile space without bypass guidance?",
    "3.2.6 Are shops placing goods or furniture on the tactile paving or pedestrian area?"
]
question_ids = [q.split(" ", 1)[0] for q in questions]

# ========== GPT 图像分析函数 ==========
def analyze_image_with_gpt4o(image_path, questions):
    with open(image_path, "rb") as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode("utf-8")

    prompt = (
        "You are an expert in urban accessibility and tactile paving assessment. "
        "Your task is to evaluate the uploaded image and provide a response to 40 predefined questions. "
        "No matter what, you MUST output exactly 40 lines, even if the image is blurry or unclear. "
        "Each line must follow the format: 'QID: 0' or 'QID: 1', where QID is the question number. "
        "Do NOT include any explanations, apologies, or comments like 'I cannot see clearly'. "
        "Just give your best judgment based on what you can observe.\n\n"
        + "\n".join(questions)
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{image_base64}",
                    "detail": "auto"
                }}
            ]
        }],
        temperature=0,
        max_tokens=1500
    )

    content = response.choices[0].message.content
    print("GPT Response:\n", content)

    # 提取 QID 和 0/1 答案
    pairs = re.findall(r'(\d+\.\d+\.\d+)\s*:\s*(0|1)', content)
    answer_dict = dict(pairs)
    answers = [int(answer_dict.get(qid, -1)) for qid in question_ids]

    print(f"✅ Extracted {len(pairs)} valid answers.")
    return answers, content

# ========== 遍历图像 ==========
results = []
image_names = []
failures = []

for filename in os.listdir(IMAGE_FOLDER):
    if filename.lower().endswith(".jpg"):
        image_path = os.path.join(IMAGE_FOLDER, filename)
        print(f"\n🔍 Processing {filename}")
        try:
            answers, raw_text = analyze_image_with_gpt4o(image_path, questions)

            # 保存 GPT 原文输出
            txt_path = os.path.join(IMAGE_FOLDER, f"{os.path.splitext(filename)[0]}_gpt_output.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(raw_text.strip())

            if -1 in answers:
                print(f"⚠️ Missing answers in {filename}")
                failures.append(filename)
                continue

            results.append(answers)
            image_names.append(filename)

        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
            failures.append(filename)

# ========== 保存结果 ==========
df = pd.DataFrame(results, columns=question_ids)
df.insert(0, "Image", image_names)

# Excel 和 CSV
df.to_excel(OUTPUT_EXCEL, index=False)
output_csv = os.path.splitext(OUTPUT_EXCEL)[0] + ".csv"
df.to_csv(output_csv, index=False, encoding="utf-8-sig")

print(f"\n✅ Excel saved to: {OUTPUT_EXCEL}")
print(f"✅ CSV saved to: {output_csv}")
if failures:
    print("⚠️ Failed or incomplete images:", failures)

# ========== 可视化表格 ==========
if df.empty:
    print("⚠️ No data to display.")
else:
    fig, ax = plt.subplots(figsize=(18, len(df) * 0.5 + 2))
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
    table.scale(1.2, 1.2)
    plt.tight_layout()
    plt.show()
