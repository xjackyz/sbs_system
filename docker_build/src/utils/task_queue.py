import asyncio
from collections import deque
import threading
from concurrent.futures import ThreadPoolExecutor
import time
from typing import Callable, Any, Dict
from dataclasses import dataclass
from datetime import datetime

from src.utils.logger import setup_logger

logger = setup_logger('task_queue')

@dataclass
class Task:
    """任务数据类"""
    id: str
    func: Callable
    args: tuple
    kwargs: dict
    priority: int
    timestamp: float
    result: Any = None
    status: str = 'pending'  # pending, running, completed, failed
    error: str = None

class AsyncTaskQueue:
    def __init__(self, max_workers=4, queue_size=100):
        """
        初始化异步任务队列
        
        Args:
            max_workers: 最大工作线程数
            queue_size: 队列最大长度
        """
        self.queue = deque(maxlen=queue_size)
        self.results = {}
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.loop = asyncio.new_event_loop()
        self.processing = False
        self.lock = threading.Lock()
        
        # 启动处理线程
        self.process_thread = threading.Thread(target=self._run_event_loop)
        self.process_thread.daemon = True
        self.process_thread.start()
        
        logger.info(f"Task queue initialized with {max_workers} workers")

    def _run_event_loop(self):
        """运行事件循环"""
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    async def _process_task(self, task: Task):
        """处理单个任务"""
        try:
            task.status = 'running'
            future = self.loop.run_in_executor(
                self.executor,
                task.func,
                *task.args,
                **task.kwargs
            )
            task.result = await future
            task.status = 'completed'
            logger.info(f"Task {task.id} completed successfully")
            
        except Exception as e:
            task.status = 'failed'
            task.error = str(e)
            logger.error(f"Task {task.id} failed: {e}")
            
        finally:
            self.results[task.id] = {
                'status': task.status,
                'result': task.result,
                'error': task.error,
                'completion_time': datetime.now().isoformat()
            }

    def add_task(self, func: Callable, args: tuple = (), kwargs: dict = None,
                 priority: int = 0) -> str:
        """
        添加任务到队列
        
        Args:
            func: 要执行的函数
            args: 位置参数
            kwargs: 关键字参数
            priority: 优先级（数字越大优先级越高）
            
        Returns:
            str: 任务ID
        """
        if kwargs is None:
            kwargs = {}
            
        task_id = f"task_{int(time.time() * 1000)}_{len(self.queue)}"
        task = Task(
            id=task_id,
            func=func,
            args=args,
            kwargs=kwargs,
            priority=priority,
            timestamp=time.time()
        )
        
        with self.lock:
            # 根据优先级插入队列
            if not self.queue:
                self.queue.append(task)
            else:
                for i, t in enumerate(self.queue):
                    if task.priority > t.priority:
                        self.queue.insert(i, task)
                        break
                else:
                    self.queue.append(task)
        
        # 触发任务处理
        asyncio.run_coroutine_threadsafe(self._process_tasks(), self.loop)
        
        logger.info(f"Task {task_id} added to queue with priority {priority}")
        return task_id

    async def _process_tasks(self):
        """处理队列中的任务"""
        if self.processing:
            return
            
        self.processing = True
        try:
            while self.queue:
                with self.lock:
                    task = self.queue.popleft()
                await self._process_task(task)
                
        finally:
            self.processing = False

    def get_task_status(self, task_id: str) -> Dict:
        """获取任务状态"""
        return self.results.get(task_id, {'status': 'unknown'})

    def clear_completed_tasks(self, max_age: int = 3600):
        """清理已完成的任务"""
        current_time = time.time()
        with self.lock:
            for task_id in list(self.results.keys()):
                result = self.results[task_id]
                if result['status'] in ['completed', 'failed']:
                    completion_time = datetime.fromisoformat(
                        result['completion_time']
                    ).timestamp()
                    if current_time - completion_time > max_age:
                        del self.results[task_id]

    def shutdown(self):
        """关闭任务队列"""
        self.executor.shutdown(wait=True)
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.process_thread.join()
        logger.info("Task queue shutdown completed") 