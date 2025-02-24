"""
Discord交互模块
"""
import os
import asyncio
from typing import Dict, Optional, Any, List
from datetime import datetime, timedelta
import discord
from discord.ext import commands
from .base import InputSource, DataValidator, DataProcessor
from ..config.input_config import DiscordConfig

class DiscordInputHandler(InputSource):
    """Discord交互处理器"""
    
    def __init__(self, config: Optional[DiscordConfig] = None):
        """初始化"""
        super().__init__()
        self.config = config or DiscordConfig()
        self.bot = commands.Bot(command_prefix=self.config.prefix)
        self.message_queue = asyncio.Queue()
        self.cooldowns = {}
        self._setup_commands()
        
    def _setup_commands(self):
        """设置命令"""
        
        @self.bot.event
        async def on_ready():
            """Bot就绪事件"""
            print(f"Discord bot已登录: {self.bot.user.name}")
            self.running = True
            
        @self.bot.command(name='analyze')
        async def analyze(ctx):
            """分析命令"""
            try:
                # 检查冷却时间
                if not self._check_cooldown(ctx.author.id, 'analyze'):
                    await ctx.send("请稍后再试")
                    return
                    
                # 检查附件
                if not ctx.message.attachments:
                    await ctx.send("请附带图片")
                    return
                    
                # 处理每个附件
                for attachment in ctx.message.attachments:
                    # 验证文件格式
                    if not self._validate_file(attachment.filename):
                        await ctx.send(f"不支持的文件格式: {attachment.filename}")
                        continue
                        
                    # 下载图片
                    image_data = await attachment.read()
                    
                    # 保存图片
                    save_path = self._save_image(image_data, attachment.filename)
                    
                    # 将消息加入队列
                    await self.message_queue.put({
                        'type': 'analyze',
                        'image': save_path,
                        'channel_id': ctx.channel.id,
                        'user_id': ctx.author.id,
                        'timestamp': datetime.now()
                    })
                    
                    await ctx.send("分析请求已接收，处理中...")
                    
            except Exception as e:
                print(f"处理分析命令出错: {e}")
                await ctx.send("处理请求时出错")
                
        @self.bot.command(name='help')
        async def help_command(ctx, command: str = None):
            """帮助命令"""
            try:
                if command:
                    # 显示特定命令的帮助
                    help_text = self._get_command_help(command)
                    if help_text:
                        await ctx.send(embed=self._create_help_embed(command, help_text))
                    else:
                        await ctx.send(f"未找到命令: {command}")
                else:
                    # 显示所有命令的帮助
                    await ctx.send(embed=self._create_help_embed())
                    
            except Exception as e:
                print(f"处理帮助命令出错: {e}")
                await ctx.send("获取帮助信息时出错")
                
        @self.bot.command(name='status')
        @commands.has_any_role(*self.config.admin_roles)
        async def status(ctx):
            """状态命令"""
            try:
                status_info = self.get_status()
                await ctx.send(embed=self._create_status_embed(status_info))
                
            except Exception as e:
                print(f"处理状态命令出错: {e}")
                await ctx.send("获取状态信息时出错")
                
        # 错误处理
        @self.bot.event
        async def on_command_error(ctx, error):
            if isinstance(error, commands.MissingAnyRole):
                await ctx.send("您没有权限执行此命令")
            elif isinstance(error, commands.CommandOnCooldown):
                await ctx.send(f"命令冷却中，请在 {error.retry_after:.1f} 秒后重试")
            else:
                print(f"命令执行出错: {error}")
                await ctx.send("命令执行出错")
                
    async def start(self):
        """启动Discord处理器"""
        await super().start()
        # 启动bot
        asyncio.create_task(self.bot.start(self.config.token))
        
    async def stop(self):
        """停止Discord处理器"""
        await self.bot.close()
        await super().stop()
        
    async def get_data(self) -> Optional[Dict[str, Any]]:
        """获取数据"""
        try:
            # 从队列中获取消息
            message_data = await self.message_queue.get()
            
            # 处理消息数据
            processed_data = DataProcessor.process_discord_data(message_data)
            
            # 更新状态
            self.last_update = datetime.now()
            
            return processed_data
            
        except Exception as e:
            print(f"获取Discord数据出错: {e}")
            return None
            
    def _check_cooldown(self, user_id: int, command: str) -> bool:
        """检查命令冷却"""
        now = datetime.now()
        cooldown_key = f"{user_id}_{command}"
        
        if cooldown_key in self.cooldowns:
            last_use = self.cooldowns[cooldown_key]
            if now - last_use < timedelta(seconds=self.config.command_cooldown):
                return False
                
        self.cooldowns[cooldown_key] = now
        return True
        
    def _validate_file(self, filename: str) -> bool:
        """验证文件格式"""
        ext = os.path.splitext(filename)[1].lower()
        return ext in self.config.allowed_formats
        
    def _save_image(self, image_data: bytes, filename: str) -> str:
        """保存图片"""
        # 确保目录存在
        os.makedirs(self.config.save_dir, exist_ok=True)
        
        # 生成保存路径
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_name = f"{timestamp}_{filename}"
        save_path = os.path.join(self.config.save_dir, save_name)
        
        # 保存文件
        with open(save_path, 'wb') as f:
            f.write(image_data)
            
        return save_path
        
    def _get_command_help(self, command: str) -> Optional[Dict]:
        """获取命令帮助"""
        help_info = {
            'analyze': {
                'description': '分析图表图片',
                'usage': f'{self.config.prefix}analyze [附带图片]',
                'example': '上传图片并输入 !analyze'
            },
            'help': {
                'description': '显示帮助信息',
                'usage': f'{self.config.prefix}help [命令名]',
                'example': '!help analyze'
            },
            'status': {
                'description': '显示系统状态（仅管理员）',
                'usage': f'{self.config.prefix}status',
                'example': '!status'
            }
        }
        return help_info.get(command)
        
    def _create_help_embed(self, command: str = None, help_info: Dict = None) -> discord.Embed:
        """创建帮助信息嵌入消息"""
        if command and help_info:
            # 特定命令的帮助
            embed = discord.Embed(
                title=f"命令帮助: {command}",
                color=discord.Color.blue()
            )
            embed.add_field(name="描述", value=help_info['description'], inline=False)
            embed.add_field(name="用法", value=help_info['usage'], inline=False)
            embed.add_field(name="示例", value=help_info['example'], inline=False)
        else:
            # 所有命令的帮助
            embed = discord.Embed(
                title="可用命令",
                description="使用 !help <命令名> 获取详细信息",
                color=discord.Color.blue()
            )
            for cmd, info in self._get_command_help('all').items():
                embed.add_field(name=cmd, value=info['description'], inline=False)
                
        return embed
        
    def _create_status_embed(self, status_info: Dict) -> discord.Embed:
        """创建状态信息嵌入消息"""
        embed = discord.Embed(
            title="系统状态",
            color=discord.Color.green() if status_info['running'] else discord.Color.red()
        )
        
        # 添加状态信息
        embed.add_field(name="运行状态", value="运行中" if status_info['running'] else "已停止")
        embed.add_field(name="最后更新", value=status_info['last_update'].strftime("%Y-%m-%d %H:%M:%S"))
        embed.add_field(name="消息队列", value=str(self.message_queue.qsize()))
        
        return embed
        
    def get_status(self) -> Dict[str, Any]:
        """获取状态"""
        status = super().get_status()
        status.update({
            'queue_size': self.message_queue.qsize(),
            'connected_servers': len(self.bot.guilds) if self.bot else 0
        })
        return status 