from setuptools import setup, find_packages

setup(
    name="sbs_system",
    version="0.1.0",
    description="SBS (Sequence Based Signal) Trading System",
    author="SBS Team",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.37.0",
        "accelerate>=0.27.0",
        "safetensors>=0.4.0",
        "sentencepiece>=0.1.99",
        "Pillow>=10.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "pyyaml>=6.0.0",
        "tqdm>=4.65.0",
        "cachetools>=5.3.0",
        "python-dotenv>=1.0.0",
        "flash-attn>=2.0.0",
        "triton>=2.0.0",
        "deepspeed>=0.10.0",
        "bitsandbytes>=0.41.0",
        "ninja>=1.11.0",
        "psutil>=5.9.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Financial and Insurance Industry',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
    ],
) 