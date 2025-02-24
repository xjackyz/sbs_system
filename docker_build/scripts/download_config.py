import requests
import yaml
import os

def download_config():
    """下载并保存Clash配置"""
    url = "http://106.75.141.168:3389/api/v1/client/subscribe?token=f65cffdc9096b82f55a4ff6af4b03d97&flag=clash"
    
    try:
        # 下载配置
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            # 确保配置目录存在
            os.makedirs('config', exist_ok=True)
            
            # 保存原始配置
            with open('config/config.yaml', 'w', encoding='utf-8') as f:
                f.write(response.text)
            print("配置下载成功")
            return True
        else:
            print(f"下载失败: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"下载出错: {e}")
        return False

if __name__ == '__main__':
    download_config() 