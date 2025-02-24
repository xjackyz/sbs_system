"""数据库连接和管理模块"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.engine import Engine
from typing import Optional
import logging
from pathlib import Path
import os

# 创建基类
Base = declarative_base()

class DatabaseManager:
    def __init__(self, config):
        """初始化数据库管理器
        
        Args:
            config: 数据库配置对象
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.engine: Optional[Engine] = None
        self.SessionLocal = None
        
    def init_db(self):
        """初始化数据库连接"""
        try:
            # 获取数据库URL
            db_url = self._get_database_url()
            
            # 创建引擎
            self.engine = create_engine(
                db_url,
                echo=self.config.debug,
                pool_size=self.config.database.pool_size,
                max_overflow=self.config.database.max_overflow
            )
            
            # 创建会话工厂
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )
            
            # 创建表
            Base.metadata.create_all(self.engine)
            
            self.logger.info("数据库初始化成功")
            return True
            
        except Exception as e:
            self.logger.error(f"数据库初始化失败: {str(e)}")
            return False
            
    def _get_database_url(self) -> str:
        """获取数据库URL"""
        db_type = self.config.database.type.lower()
        
        if db_type == 'postgresql':
            return (f"postgresql://{self.config.database.user}:"
                   f"{self.config.database.password}@"
                   f"{self.config.database.host}:"
                   f"{self.config.database.port}/"
                   f"{self.config.database.name}")
        elif db_type == 'sqlite':
            # 确保数据目录存在
            db_path = Path(self.config.database.path)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            return f"sqlite:///{db_path}"
        else:
            raise ValueError(f"不支持的数据库类型: {db_type}")
            
    def get_session(self):
        """获取数据库会话"""
        if not self.SessionLocal:
            raise Exception("数据库未初始化")
        return self.SessionLocal()
    
    def cleanup(self):
        """清理资源"""
        if self.engine:
            self.engine.dispose()
            
    def __enter__(self):
        """上下文管理器入口"""
        return self.get_session()
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        if exc_type:
            self.logger.error(f"数据库操作出错: {str(exc_val)}")
        if self.SessionLocal:
            self.SessionLocal.close_all() 