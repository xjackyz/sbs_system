import json
import os

# 需要处理的文件列表
files = [
    'llava_training_breakout_data.json',
    'llava_training_DTDB_data.json',
    'llava_training_liquidate_data.json',
    'llava_training_sce_data.json',
    'llava_training_sbs_label_data.json',
    'llava_training_sma_data.json',
    'llava_training_swing_data.json',
    'llava_training_TPSL_data.json'
]

def add_png_extension(file_path):
    try:
        # 读取JSON文件
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 检查是否是列表类型
        if isinstance(data, list):
            # 遍历所有条目并修改image路径
            for item in data:
                if 'image' in item and not item['image'].endswith('.png'):
                    item['image'] = item['image'] + '.png'
        
        # 将修改后的数据写回文件
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        
        print(f"✓ {file_path} 处理完成")
        
    except FileNotFoundError:
        print(f"× {file_path} 文件不存在，已跳过")
    except Exception as e:
        print(f"× {file_path} 处理出错: {str(e)}")

# 处理所有文件
for file in files:
    add_png_extension(file)

print("\n所有文件处理完成！")