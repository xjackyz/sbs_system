import os
import requests
import json
import yaml
from pathlib import Path

def configure_v2ray():
    """配置v2ray"""
    try:
        # 下载订阅内容
        url = "http://106.75.141.168:3389/api/v1/client/subscribe?token=f65cffdc9096b82f55a4ff6af4b03d97&flag=clash"
        response = requests.get(url, timeout=30)
        
        if response.status_code != 200:
            print(f"下载配置失败: {response.status_code}")
            return False
            
        # 解析Clash配置
        clash_config = response.text
        
        # 转换为v2ray配置
        v2ray_config = {
            "inbounds": [{
                "port": 1080,
                "protocol": "socks",
                "settings": {
                    "auth": "noauth",
                    "udp": True
                }
            }],
            "outbounds": []
        }
        
        # 解析Clash配置中的代理节点
        clash_data = yaml.safe_load(clash_config)
        proxies = clash_data.get('proxies', [])
        
        # 添加trojan节点
        for proxy in proxies:
            if proxy['type'] == 'trojan':
                outbound = {
                    "protocol": "trojan",
                    "settings": {
                        "servers": [{
                            "address": proxy['server'],
                            "port": proxy['port'],
                            "password": proxy['password'],
                            "sni": proxy.get('sni', '')
                        }]
                    }
                }
                v2ray_config['outbounds'].append(outbound)
        
        # 保存v2ray配置
        config_file = '/usr/local/etc/v2ray/config.json'
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        
        with open(config_file, 'w') as f:
            json.dump(v2ray_config, f, indent=2)
            
        print("配置文件已保存到", config_file)
        print("\n使用方法:")
        print("1. 启动v2ray: sudo systemctl start v2ray")
        print("2. 设置开机自启: sudo systemctl enable v2ray")
        print("3. 查看状态: sudo systemctl status v2ray")
        
        return True
        
    except Exception as e:
        print(f"配置出错: {e}")
        return False

if __name__ == '__main__':
    configure_v2ray() 