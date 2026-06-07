import os
import ast
from openai import OpenAI
from config import ARK_API_KEY, ARK_BASE_URL, ARK_TEXT_ENDPOINT

def ctx_creator(req, text):
    client = OpenAI(
        api_key=ARK_API_KEY,
        base_url=ARK_BASE_URL
    )
    
    messages = [
        {"role": "system", "content": f"输入内容里面的duration的值为当段视频内容的时长，按照片段视频的时长计算总时长，将输入内容按{req}的需求，编排成高质量的视频脚本，并输出对应的视频片段的uuid数组"},
        {"role": "user", "content": text}
    ]

    response = client.chat.completions.create(
        model=ARK_TEXT_ENDPOINT,
        messages=messages,
        temperature=0.7
    )
    
    print("response:", response)
    res = response.choices[0].message.content
    print(res)
    return res

if __name__ == '__main__':
    # 从 ./data/reformat_data.json 读取内容
    with open('./data/reformat_data.json', 'r', encoding='utf-8') as f:
        text = f.read()
    req = """要一个海滩旅游的短视频，视频总时长需要120秒，希望有记忆点以及有对应的开头高潮起伏，使得视频观看者更容易留下持续观看。"""
    res = ctx_creator(req, text)
    # 将结果写入 ./data/story.txt
    with open('./data/story.txt', 'w', encoding='utf-8') as f:
        f.write(str(res))
    print("end")
