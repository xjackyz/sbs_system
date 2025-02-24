import logging
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
import json
import torch
from src.model.sbs_analyzer import SBSSteps

class TradingAlertManager:
    def __init__(self, config_path: Optional[str] = None):
        """初始化交易提醒管理器
        
        Args:
            config_path: 配置文件路径
        """
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path) if config_path else {}
        self.alerts_history = []
        
        # 创建保存目录
        self.save_dir = Path(self.config.get('save_dir', 'alerts'))
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置"""
        with open(config_path, 'r') as f:
            return json.load(f)
            
    def generate_alert(self,
                      step: SBSSteps,
                      confidence: float,
                      symbol: str,
                      timeframe: str) -> Dict[str, Any]:
        """生成交易提醒
        
        Args:
            step: 预测的SBS步骤
            confidence: 预测置信度
            symbol: 交易对
            timeframe: 时间周期
        """
        alert = {
            'timestamp': datetime.now().isoformat(),
            'step': step.name,
            'confidence': confidence,
            'symbol': symbol,
            'timeframe': timeframe,
            'action': self._get_trading_action(step)
        }
        
        # 记录提醒
        self.alerts_history.append(alert)
        
        # 保存提醒
        self._save_alert(alert)
        
        # 发送通知
        self._send_notification(alert)
        
        return alert
    
    def _get_trading_action(self, step: SBSSteps) -> str:
        """根据SBS步骤获取交易动作"""
        actions = {
            SBSSteps.SETUP: "准备建仓",
            SBSSteps.BUILDUP: "加仓",
            SBSSteps.STRENGTH: "持有",
            SBSSteps.BREAKOUT: "突破确认",
            SBSSteps.SELL: "卖出"
        }
        return actions.get(step, "观察")
    
    def _save_alert(self, alert: Dict[str, Any]):
        """保存交易提醒"""
        # 生成文件名
        filename = f"alert_{alert['timestamp'].replace(':', '-')}.json"
        
        # 保存为JSON文件
        with open(self.save_dir / filename, 'w') as f:
            json.dump(alert, f, indent=4)
            
    def _send_notification(self, alert: Dict[str, Any]):
        """发送通知
        
        可以根据需要实现不同的通知方式：
        - 钉钉
        - 企业微信
        - 电子邮件
        - Telegram
        等
        """
        message = (
            f"交易提醒\n"
            f"时间: {alert['timestamp']}\n"
            f"交易对: {alert['symbol']}\n"
            f"周期: {alert['timeframe']}\n"
            f"步骤: {alert['step']}\n"
            f"置信度: {alert['confidence']:.2f}\n"
            f"建议动作: {alert['action']}"
        )
        
        self.logger.info(message)
        
        # TODO: 实现具体的通知方式
        
    def get_alerts_history(self,
                          symbol: Optional[str] = None,
                          start_time: Optional[datetime] = None,
                          end_time: Optional[datetime] = None) -> list:
        """获取历史提醒
        
        Args:
            symbol: 交易对
            start_time: 开始时间
            end_time: 结束时间
        """
        filtered_alerts = self.alerts_history
        
        if symbol:
            filtered_alerts = [
                alert for alert in filtered_alerts
                if alert['symbol'] == symbol
            ]
            
        if start_time:
            filtered_alerts = [
                alert for alert in filtered_alerts
                if datetime.fromisoformat(alert['timestamp']) >= start_time
            ]
            
        if end_time:
            filtered_alerts = [
                alert for alert in filtered_alerts
                if datetime.fromisoformat(alert['timestamp']) <= end_time
            ]
            
        return filtered_alerts
    
    def analyze_alerts(self, alerts: list) -> Dict[str, Any]:
        """分析交易提醒
        
        Args:
            alerts: 交易提醒列表
        """
        if not alerts:
            return {}
            
        # 统计各个步骤的数量
        step_counts = {}
        for alert in alerts:
            step = alert['step']
            step_counts[step] = step_counts.get(step, 0) + 1
            
        # 计算平均置信度
        avg_confidence = sum(alert['confidence'] for alert in alerts) / len(alerts)
        
        # 获取时间范围
        timestamps = [datetime.fromisoformat(alert['timestamp']) for alert in alerts]
        time_range = max(timestamps) - min(timestamps)
        
        return {
            'total_alerts': len(alerts),
            'step_distribution': step_counts,
            'average_confidence': avg_confidence,
            'time_range_hours': time_range.total_seconds() / 3600,
            'first_alert': min(timestamps).isoformat(),
            'last_alert': max(timestamps).isoformat()
        }
        
    def clear_history(self):
        """清除历史记录"""
        self.alerts_history = [] 