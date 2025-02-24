# 贡献指南

感谢你考虑为 SBS Trading System 做出贡献！

## 行为准则

本项目采用 [Contributor Covenant](https://www.contributor-covenant.org/version/2/0/code_of_conduct/) 行为准则。请确保你的所有互动都符合这些准则。

## 如何贡献

### 报告 Bug

1. 使用 GitHub Issues 搜索确认该 bug 尚未被报告
2. 如果找不到相关 issue，创建一个新的
3. 清晰地描述问题，包括：
   - 复现步骤
   - 预期行为
   - 实际行为
   - 系统环境信息
   - 相关日志或截图

### 提交代码

1. Fork 本仓库
2. 创建你的特性分支：`git checkout -b feature/my-new-feature`
3. 提交你的改动：`git commit -am 'Add some feature'`
4. 推送到分支：`git push origin feature/my-new-feature`
5. 提交 Pull Request

### 代码规范

- 遵循 PEP 8 规范
- 使用类型注解
- 添加适当的注释和文档字符串
- 确保通过所有测试
- 新功能需要添加相应的测试用例

### 提交信息规范

使用清晰的提交信息，格式如下：

```
<type>(<scope>): <subject>

<body>

<footer>
```

类型（type）：
- feat: 新功能
- fix: 修复
- docs: 文档
- style: 格式
- refactor: 重构
- test: 测试
- chore: 构建过程或辅助工具的变动

### 文档贡献

- 确保文档清晰易懂
- 更新相关的 API 文档
- 添加必要的示例代码
- 修正任何发现的错误

## 开发流程

1. 安装依赖：
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

2. 运行测试：
```bash
python -m pytest tests/
```

3. 检查代码风格：
```bash
flake8 src/
mypy src/
```

## 发布流程

1. 更新版本号（遵循语义化版本）
2. 更新 CHANGELOG.md
3. 创建发布标签
4. 推送到 GitHub

## 获取帮助

如果你需要帮助，可以：

1. 查看项目文档
2. 创建 Issue
3. 发送邮件到 xjackyz@gmail.com

## 感谢

再次感谢你的贡献！ 