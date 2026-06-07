import os
import json
import base64
from openai import OpenAI
from config import ARK_API_KEY, ARK_BASE_URL, ARK_VISION_ENDPOINT
from get_duration import get_video_duration_func
import cv2

def extract_frames_base64(video_path, fps=1):
    """从视频中抽取帧并转为base64编码"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("无法打开视频文件")
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps)
    frames_base64 = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            _, buffer = cv2.imencode('.jpg', frame)
            base64_str = base64.b64encode(buffer).decode('utf-8')
            frames_base64.append(f"data:image/jpeg;base64,{base64_str}")
        
        frame_count += 1
    
    cap.release()
    return frames_base64

def video_understand_func(video_path, duration):
    client = OpenAI(
        api_key=ARK_API_KEY,
        base_url=ARK_BASE_URL
    )
    
    # 抽取视频帧
    frames = extract_frames_base64(video_path, fps=1)
    if not frames:
        raise ValueError("无法从视频中抽取帧")
    
    # 构建消息
    content = []
    for i, frame_base64 in enumerate(frames):
        content.append({
            "type": "image_url",
            "image_url": {"url": frame_base64}
        })
    
    content.append({
        "type": "text",
        "text": f"""这段视频时长为{duration}秒，包含{len(frames)}帧。请按时间顺序分析每个时间段的视频内容和运镜风格，输出格式如下：
#### 时间范围，格式:0s-3s
**视频内容：**
**运镜风格：**"""
    })
    
    messages = [
        {
            "role": "system",
            "content": "你是一位视频内容和运镜理解专家，擅长按时间先后顺序分析视频片段的内容和拍摄风格。"
        },
        {
            "role": "user",
            "content": content
        }
    ]
    
    response = client.chat.completions.create(
        model=ARK_VISION_ENDPOINT,
        messages=messages,
        temperature=0
    )
    
    return response.choices[0].message.content

def main():
    output_dir = "/Users/elvis/workspace/code/ai-cutting/video_part/"  # 拆分后的片段目录
    data_dir = "./data"
    os.makedirs(data_dir, exist_ok=True)
    understand_json_path = os.path.join(data_dir, "understand.txt")
    res_list = []

    # 遍历所有子目录和文件，找到所有mp4文件
    for root, dirs, files in os.walk(output_dir):
        for filename in files:
            if filename.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
                video_path = os.path.join(root, filename)
                duration = get_video_duration_func(video_path)  # 获取视频时长

                log_msg = f"理解视频内容 starting for video: {video_path} with duration: {duration} seconds"
                print(log_msg)
                try:
                    res = video_understand_func(video_path, duration)
                    print(f"视频理解结果: {res}")
                    # 将log和res一起写入
                    res_list.append({
                        "log": log_msg,
                        "result": res
                    })
                    # 每次追加后写入，保证并发和时序性
                    with open(understand_json_path, 'w', encoding='utf-8') as f:
                        json.dump(res_list, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    print(f"理解视频内容时发生错误: {e}")

if __name__ == "__main__":
    main()
