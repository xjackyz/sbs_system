"""LLaVA分析器模块"""
import os
import logging
import warnings
from pathlib import Path
import ssl
import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning
from urllib3.util import ssl_

# 禁用 SSL 警告
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context

# 创建一个不验证SSL的session
session = requests.Session()
session.verify = False

# 设置环境变量
os.environ.update({
    'TRANSFORMERS_OFFLINE': '0',  # 允许在线下载
    'HF_HUB_OFFLINE': '0',       # 允许在线下载
    'USE_TORCH': '1',            # 使用 PyTorch
    'TRANSFORMERS_VERBOSITY': 'error',  # 只显示错误日志
    'CURL_CA_BUNDLE': '',        # 禁用 SSL 验证
    'REQUESTS_CA_BUNDLE': '',    # 禁用 SSL 验证
    'SSL_CERT_FILE': ''         # 禁用 SSL 验证
})

# 使用环境变量中的代理设置
proxy = os.getenv('HTTP_PROXY') or os.getenv('HTTPS_PROXY')
if proxy:
    os.environ.update({
        'http_proxy': proxy,
        'https_proxy': proxy,
        'TRANSFORMERS_HTTP_PROXY': proxy,
        'HF_HUB_PROXY': proxy
    })

# 设置缓存目录
cache_dir = os.path.expanduser('~/.cache/huggingface')
os.environ.update({
    'HF_HOME': cache_dir,
    'TRANSFORMERS_CACHE': os.path.join(cache_dir, 'transformers'),
    'HF_HUB_CACHE': os.path.join(cache_dir, 'hub'),
    'TORCH_HOME': os.path.join(cache_dir, 'torch'),
    'XDG_CACHE_HOME': os.path.expanduser('~/.cache')
})

# 创建缓存目录
for path in [os.environ['HF_HOME'], os.environ['TRANSFORMERS_CACHE'], 
            os.environ['HF_HUB_CACHE'], os.environ['TORCH_HOME']]:
    os.makedirs(path, exist_ok=True)

import torch
from torch import cuda
from PIL import Image
from typing import Dict, Any, Optional, List
from transformers import CLIPVisionModel, LlamaForCausalLM, CLIPImageProcessor, LlamaTokenizerFast
from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
from ..config import Config

logger = logging.getLogger(__name__)

def check_model_files(model_path: str) -> bool:
    """检查模型文件是否存在。

    Args:
        model_path: 模型路径

    Returns:
        bool: 是否存在所有必需的文件
    """
    required_files = [
        'config.json',
        'tokenizer_config.json',
        'tokenizer.model',
        'model.safetensors.index.json'
    ]
    model_dir = Path(model_path)
    return all((model_dir / f).exists() for f in required_files)

class LLaVAAnalyzer:
    def __init__(self, config: Config):
        """初始化LLaVA分析器。

        Args:
            config (Config): 配置对象
        """
        self.config = config
        self._logger = logger
        self.model = None
        self.vision_tower = None
        self.image_processor = None
        self.tokenizer = None
        self.result_cache = {}
        
        try:
            self._logger.info("正在初始化LLaVA分析器...")
            
            # 检查语言模型路径
            if not os.path.exists(self.config.model_path):
                raise ValueError(f"模型路径不存在: {self.config.model_path}")
            if not check_model_files(self.config.model_path):
                raise ValueError(f"模型文件不完整: {self.config.model_path}")
            
            # 设置SSL和代理配置
            ssl_context = ssl_.create_urllib3_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            # 设置requests的SSL验证
            requests.packages.urllib3.disable_warnings()
            session = requests.Session()
            session.verify = False
            
            # 设置代理
            proxy = os.environ.get('HTTP_PROXY') or os.environ.get('HTTPS_PROXY')
            if proxy:
                session.proxies = {
                    'http': proxy,
                    'https': proxy
                }
            
            # 设置模型加载参数
            llm_kwargs = {
                'local_files_only': False,
                'trust_remote_code': True,
                'cache_dir': self.config.cache_dir
            }
            
            vision_kwargs = {
                'local_files_only': False,
                'trust_remote_code': True,
                'cache_dir': self.config.cache_dir,
                'verify': False,  # 禁用SSL验证
                'force_download': True  # 强制重新下载
            }
            
            # 加载图像处理器
            self._logger.info("正在加载图像处理器...")
            try:
                self.image_processor = CLIPImageProcessor(
                    crop_size=224,  # 根据image_crop_resolution设置
                    do_center_crop=True,
                    do_normalize=True,
                    do_resize=True,
                    image_mean=[0.48145466, 0.4578275, 0.40821073],
                    image_std=[0.26862954, 0.26130258, 0.27577711],
                    resample=3,
                    size=224  # 根据image_split_resolution设置
                )
            except Exception as e:
                self._logger.error(f"加载图像处理器失败: {str(e)}")
                raise e
            
            # 加载视觉模型
            self._logger.info("正在加载视觉模型...")
            try:
                self.vision_tower = CLIPVisionModel.from_pretrained(
                    self.config.model_path,
                    local_files_only=True,
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    use_safetensors=True,
                    mm_vision_select_layer=-2,  # 根据配置文件设置
                    mm_vision_select_feature="patch"  # 根据配置文件设置
                )
            except Exception as e:
                self._logger.error(f"加载视觉模型失败: {str(e)}")
                raise e
            
            # 加载语言模型
            self._logger.info("正在加载语言模型...")
            try:
                self.model = LlamaForCausalLM.from_pretrained(
                    self.config.model_path,
                    local_files_only=True,
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    use_safetensors=True
                )
            except Exception as e:
                self._logger.error(f"加载语言模型失败: {str(e)}")
                raise e
            
            # 加载分词器
            self._logger.info("正在加载分词器...")
            try:
                self.tokenizer = LlamaTokenizerFast.from_pretrained(
                    self.config.model_path,
                    local_files_only=True,
                    trust_remote_code=True,
                    use_safetensors=True
                )
            except Exception as e:
                self._logger.error(f"加载分词器失败: {str(e)}")
                raise e
            
            self._logger.info("LLaVA分析器初始化完成")
            
        except Exception as e:
            self._logger.error(f"LLaVA分析器初始化失败: {str(e)}")
            self._logger.error("错误详情:", exc_info=True)
            raise
            
    def analyze_image(self, image_path: str, prompt: str) -> str:
        """分析图片。

        Args:
            image_path: 图片路径
            prompt: 提示文本

        Returns:
            str: 分析结果
        """
        try:
            # 读取图像
            self._logger.info(f"正在读取图像: {image_path}")
            image = Image.open(image_path).convert('RGB')
            
            # 处理图像
            self._logger.info("正在处理图像...")
            image_inputs = self.image_processor(
                images=image,
                return_tensors="pt"
            ).to(self.config.device)
            
            # 获取图像特征
            with torch.no_grad():
                image_outputs = self.vision_tower(
                    image_inputs.pixel_values,
                    output_hidden_states=True
                )
                image_features = image_outputs.hidden_states[self.config.mm_vision_select_layer]
                
                # 根据配置选择特征
                if self.config.mm_vision_select_feature == "patch":
                    image_features = image_features[:, 1:]  # 移除CLS token
                
            # 将图像特征添加到模型的状态中
            self.model.vision_hidden_states = image_features
            
            # 处理文本
            self._logger.info("正在处理文本...")
            # 确保prompt不为空
            if not prompt:
                prompt = self.config.default_prompt
                
            text_inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_length
            ).to(self.config.device)
            
            # 准备多模态输入
            inputs = {
                "input_ids": text_inputs.input_ids,
                "attention_mask": text_inputs.attention_mask,
                "image_features": image_features
            }
            
            # 使用自动混合精度进行推理
            self._logger.info("正在生成分析...")
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    # 首先将图像特征输入模型
                    self.model.prepare_inputs_for_generation(inputs)
                    outputs = self.model.generate(
                        input_ids=text_inputs.input_ids,
                        attention_mask=text_inputs.attention_mask,
                        max_length=self.config.max_length,
                        temperature=self.config.temperature,
                        top_p=self.config.top_p,
                        num_beams=self.config.num_beams,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
            
            # 解码输出
            self._logger.info("正在解码输出...")
            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            self._logger.info("分析完成")
            return result
            
        except Exception as e:
            self._logger.error(f"分析图像时出错: {str(e)}")
            self._logger.error("错误详情:", exc_info=True)
            raise
            
    def cleanup(self):
        """清理资源。"""
        try:
            # 清理缓存
            self.result_cache.clear()
            
            # 清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 删除模型和处理器
            if hasattr(self, 'model'):
                del self.model
            if hasattr(self, 'vision_tower'):
                del self.vision_tower
            if hasattr(self, 'image_processor'):
                del self.image_processor
            if hasattr(self, 'tokenizer'):
                del self.tokenizer
            
            self._logger.info("清理完成")
            
        except Exception as e:
            self._logger.error(f"清理时出错: {str(e)}")
            
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()