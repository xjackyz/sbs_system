import requests
import os
import gzip
import shutil
import stat

def download_clash():
    """下载Clash二进制文件"""
    url = "https://ghproxy.com/https://github.com/Dreamacro/clash/releases/download/v1.18.0/clash-linux-amd64-v1.18.0.gz"
    
    try:
        # 下载文件
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            # 保存gzip文件
            with open('clash.gz', 'wb') as f:
                f.write(response.content)
            
            # 解压文件
            with gzip.open('clash.gz', 'rb') as f_in:
                with open('clash', 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # 添加执行权限
            os.chmod('clash', os.stat('clash').st_mode | stat.S_IEXEC)
            
            # 移动到/usr/local/bin
            os.system('sudo mv clash /usr/local/bin/')
            
            # 清理临时文件
            if os.path.exists('clash.gz'):
                os.remove('clash.gz')
                
            print("Clash下载并安装成功")
            return True
        else:
            print(f"下载失败: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"下载出错: {e}")
        return False

if __name__ == '__main__':
    download_clash() 