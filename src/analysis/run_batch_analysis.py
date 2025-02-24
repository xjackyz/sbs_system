import argparse
import logging
from datetime import datetime
from batch_analyzer import BatchAnalyzer

def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='批量分析图片')
    parser.add_argument('--input_dir', required=True, help='输入目录路径')
    parser.add_argument('--model_path', required=True, help='模型路径')
    parser.add_argument('--output_format', default='json', choices=['json'], help='输出格式')
    parser.add_argument('--device', default='cuda', help='运行设备')
    args = parser.parse_args()
    
    try:
        # 初始化分析器
        analyzer = BatchAnalyzer(args.model_path, args.device)
        
        # 分析目录
        results = analyzer.analyze_directory(args.input_dir)
        
        # 生成报告
        output_file = analyzer.generate_report(results, args.output_format)
        print(f"分析完成，报告已保存到: {output_file}")
        
    except Exception as e:
        print(f"运行过程中出错: {str(e)}")
        raise

if __name__ == '__main__':
    main() 