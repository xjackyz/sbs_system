<template>
  <div class="settings">
    <el-tabs v-model="activeTab" type="border-card">
      <!-- 模型配置 -->
      <el-tab-pane label="模型配置" name="model">
        <el-form
          ref="modelForm"
          :model="modelConfig"
          :rules="modelRules"
          label-width="120px"
        >
          <el-form-item label="基础模型" prop="baseModel">
            <el-input v-model="modelConfig.baseModel" placeholder="模型路径">
              <template #append>
                <el-button @click="selectModel">选择</el-button>
              </template>
            </el-input>
          </el-form-item>

          <el-form-item label="运行设备" prop="device">
            <el-radio-group v-model="modelConfig.device">
              <el-radio label="cuda">CUDA (GPU)</el-radio>
              <el-radio label="cpu">CPU</el-radio>
            </el-radio-group>
          </el-form-item>

          <el-form-item label="最大Token数" prop="maxNewTokens">
            <el-input-number
              v-model="modelConfig.maxNewTokens"
              :min="100"
              :max="2000"
              :step="100"
            />
          </el-form-item>

          <el-form-item label="置信度阈值" prop="confidenceThreshold">
            <el-slider
              v-model="modelConfig.confidenceThreshold"
              :step="0.05"
              :min="0"
              :max="1"
              :format-tooltip="value => `${(value * 100).toFixed(0)}%`"
            />
          </el-form-item>
        </el-form>
      </el-tab-pane>

      <!-- 系统配置 -->
      <el-tab-pane label="系统配置" name="system">
        <el-form
          ref="systemForm"
          :model="systemConfig"
          :rules="systemRules"
          label-width="120px"
        >
          <el-form-item label="日志级别" prop="logLevel">
            <el-select v-model="systemConfig.logLevel">
              <el-option label="DEBUG" value="DEBUG" />
              <el-option label="INFO" value="INFO" />
              <el-option label="WARNING" value="WARNING" />
              <el-option label="ERROR" value="ERROR" />
            </el-select>
          </el-form-item>

          <el-form-item label="临时目录" prop="tempDir">
            <el-input v-model="systemConfig.tempDir">
              <template #append>
                <el-button @click="selectDirectory">选择</el-button>
              </template>
            </el-input>
          </el-form-item>

          <el-form-item label="自动清理" prop="autoClear">
            <el-switch v-model="systemConfig.autoClear" />
            <span class="setting-hint">
              自动清理超过7天的临时文件
            </span>
          </el-form-item>

          <el-form-item label="健康检查间隔" prop="healthCheckInterval">
            <el-input-number
              v-model="systemConfig.healthCheckInterval"
              :min="10"
              :max="300"
              :step="10"
            >
              <template #append>秒</template>
            </el-input-number>
          </el-form-item>
        </el-form>
      </el-tab-pane>

      <!-- 代理设置 -->
      <el-tab-pane label="代理设置" name="proxy">
        <el-form
          ref="proxyForm"
          :model="proxyConfig"
          :rules="proxyRules"
          label-width="120px"
        >
          <el-form-item label="启用代理" prop="enabled">
            <el-switch v-model="proxyConfig.enabled" />
          </el-form-item>

          <template v-if="proxyConfig.enabled">
            <el-form-item label="HTTP代理" prop="httpProxy">
              <el-input v-model="proxyConfig.httpProxy" placeholder="http://127.0.0.1:7890" />
            </el-form-item>

            <el-form-item label="HTTPS代理" prop="httpsProxy">
              <el-input v-model="proxyConfig.httpsProxy" placeholder="http://127.0.0.1:7890" />
            </el-form-item>

            <el-form-item label="SOCKS代理" prop="socksProxy">
              <el-input v-model="proxyConfig.socksProxy" placeholder="socks5://127.0.0.1:7890" />
            </el-form-item>

            <el-form-item label="不使用代理" prop="noProxy">
              <el-input
                v-model="proxyConfig.noProxy"
                type="textarea"
                :rows="2"
                placeholder="localhost,127.0.0.1"
              />
            </el-form-item>
          </template>
        </el-form>
      </el-tab-pane>

      <!-- Discord设置 -->
      <el-tab-pane label="Discord设置" name="discord">
        <el-form
          ref="discordForm"
          :model="discordConfig"
          :rules="discordRules"
          label-width="120px"
        >
          <el-form-item label="Bot Token" prop="botToken">
            <el-input
              v-model="discordConfig.botToken"
              type="password"
              show-password
              placeholder="Discord Bot Token"
            />
          </el-form-item>

          <el-form-item label="信号Webhook" prop="signalWebhook">
            <el-input
              v-model="discordConfig.signalWebhook"
              placeholder="交易信号推送Webhook URL"
            />
          </el-form-item>

          <el-form-item label="监控Webhook" prop="monitorWebhook">
            <el-input
              v-model="discordConfig.monitorWebhook"
              placeholder="系统监控Webhook URL"
            />
          </el-form-item>

          <el-form-item label="上传Webhook" prop="uploadWebhook">
            <el-input
              v-model="discordConfig.uploadWebhook"
              placeholder="图片上传Webhook URL"
            />
          </el-form-item>

          <el-form-item label="Bot头像" prop="botAvatar">
            <el-input v-model="discordConfig.botAvatar" placeholder="Bot头像URL" />
          </el-form-item>
        </el-form>
      </el-tab-pane>
    </el-tabs>

    <!-- 操作按钮 -->
    <div class="action-buttons">
      <el-button type="primary" @click="saveSettings">保存设置</el-button>
      <el-button @click="resetSettings">重置</el-button>
      <el-button type="warning" @click="restartSystem">重启系统</el-button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import axios from 'axios'

// 当前激活的标签页
const activeTab = ref('model')

// 表单引用
const modelForm = ref(null)
const systemForm = ref(null)
const proxyForm = ref(null)
const discordForm = ref(null)

// 模型配置
const modelConfig = ref({
  baseModel: '',
  device: 'cuda',
  maxNewTokens: 1000,
  confidenceThreshold: 0.8
})

// 系统配置
const systemConfig = ref({
  logLevel: 'INFO',
  tempDir: 'temp',
  autoClear: true,
  healthCheckInterval: 30
})

// 代理配置
const proxyConfig = ref({
  enabled: false,
  httpProxy: '',
  httpsProxy: '',
  socksProxy: '',
  noProxy: ''
})

// Discord配置
const discordConfig = ref({
  botToken: '',
  signalWebhook: '',
  monitorWebhook: '',
  uploadWebhook: '',
  botAvatar: ''
})

// 表单验证规则
const modelRules = {
  baseModel: [
    { required: true, message: '请输入模型路径', trigger: 'blur' }
  ],
  device: [
    { required: true, message: '请选择运行设备', trigger: 'change' }
  ]
}

const systemRules = {
  logLevel: [
    { required: true, message: '请选择日志级别', trigger: 'change' }
  ],
  tempDir: [
    { required: true, message: '请输入临时目录', trigger: 'blur' }
  ]
}

const proxyRules = {
  httpProxy: [
    { pattern: /^https?:\/\/.+/, message: '请输入有效的HTTP代理地址', trigger: 'blur' }
  ],
  httpsProxy: [
    { pattern: /^https?:\/\/.+/, message: '请输入有效的HTTPS代理地址', trigger: 'blur' }
  ],
  socksProxy: [
    { pattern: /^socks[45]:\/\/.+/, message: '请输入有效的SOCKS代理地址', trigger: 'blur' }
  ]
}

const discordRules = {
  botToken: [
    { required: true, message: '请输入Bot Token', trigger: 'blur' }
  ],
  signalWebhook: [
    { required: true, message: '请输入信号Webhook URL', trigger: 'blur' },
    { pattern: /^https:\/\/discord\.com\/api\/webhooks\/.+/, message: '请输入有效的Discord Webhook URL', trigger: 'blur' }
  ],
  monitorWebhook: [
    { required: true, message: '请输入监控Webhook URL', trigger: 'blur' },
    { pattern: /^https:\/\/discord\.com\/api\/webhooks\/.+/, message: '请输入有效的Discord Webhook URL', trigger: 'blur' }
  ],
  uploadWebhook: [
    { required: true, message: '请输入上传Webhook URL', trigger: 'blur' },
    { pattern: /^https:\/\/discord\.com\/api\/webhooks\/.+/, message: '请输入有效的Discord Webhook URL', trigger: 'blur' }
  ]
}

// 获取配置
const fetchConfig = async () => {
  try {
    const response = await axios.get('/api/config')
    const config = response.data
    
    modelConfig.value = config.model
    systemConfig.value = config.system
    proxyConfig.value = config.proxy
    discordConfig.value = config.discord
  } catch (error) {
    ElMessage.error('获取配置失败')
  }
}

// 保存配置
const saveSettings = async () => {
  try {
    // 验证表单
    await Promise.all([
      modelForm.value?.validate(),
      systemForm.value?.validate(),
      proxyForm.value?.validate(),
      discordForm.value?.validate()
    ])
    
    // 保存配置
    await axios.post('/api/config', {
      model: modelConfig.value,
      system: systemConfig.value,
      proxy: proxyConfig.value,
      discord: discordConfig.value
    })
    
    ElMessage.success('配置保存成功')
  } catch (error) {
    ElMessage.error('配置保存失败')
  }
}

// 重置配置
const resetSettings = async () => {
  try {
    await ElMessageBox.confirm('确定要重置所有配置吗？', '警告', {
      type: 'warning'
    })
    await fetchConfig()
    ElMessage.success('配置已重置')
  } catch (error) {
    if (error !== 'cancel') {
      ElMessage.error('重置配置失败')
    }
  }
}

// 重启系统
const restartSystem = async () => {
  try {
    await ElMessageBox.confirm(
      '确定要重启系统吗？这将中断所有正在进行的分析任务。',
      '警告',
      {
        type: 'warning',
        confirmButtonText: '确定重启',
        confirmButtonClass: 'el-button--danger'
      }
    )
    
    await axios.post('/api/system/restart')
    ElMessage.success('系统重启指令已发送')
  } catch (error) {
    if (error !== 'cancel') {
      ElMessage.error('发送重启指令失败')
    }
  }
}

// 选择模型
const selectModel = () => {
  // TODO: 实现模型选择对话框
}

// 选择目录
const selectDirectory = () => {
  // TODO: 实现目录选择对话框
}

onMounted(() => {
  fetchConfig()
})
</script>

<style scoped>
.settings {
  padding: 20px;
}

.setting-hint {
  margin-left: 10px;
  color: #909399;
  font-size: 12px;
}

.action-buttons {
  margin-top: 20px;
  display: flex;
  justify-content: center;
  gap: 20px;
}
</style> 