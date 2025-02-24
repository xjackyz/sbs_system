<template>
  <div class="logs">
    <!-- 过滤器 -->
    <el-card class="filter-card">
      <el-form :inline="true" :model="filterForm" class="filter-form">
        <el-form-item label="日志级别">
          <el-select
            v-model="filterForm.level"
            multiple
            collapse-tags
            placeholder="选择日志级别"
          >
            <el-option label="INFO" value="INFO" />
            <el-option label="WARNING" value="WARNING" />
            <el-option label="ERROR" value="ERROR" />
            <el-option label="DEBUG" value="DEBUG" />
          </el-select>
        </el-form-item>
        <el-form-item label="时间范围">
          <el-date-picker
            v-model="filterForm.dateRange"
            type="datetimerange"
            range-separator="至"
            start-placeholder="开始时间"
            end-placeholder="结束时间"
          />
        </el-form-item>
        <el-form-item label="关键词">
          <el-input
            v-model="filterForm.keyword"
            placeholder="搜索日志内容"
            clearable
          />
        </el-form-item>
        <el-form-item>
          <el-button type="primary" @click="handleSearch">搜索</el-button>
          <el-button @click="resetForm">重置</el-button>
        </el-form-item>
      </el-form>
    </el-card>

    <!-- 日志显示区域 -->
    <el-card class="log-card">
      <template #header>
        <div class="card-header">
          <div class="header-left">
            <span>系统日志</span>
            <el-switch
              v-model="autoScroll"
              active-text="自动滚动"
              style="margin-left: 20px"
            />
          </div>
          <div class="header-right">
            <el-button type="primary" size="small" @click="exportLogs">
              导出日志
            </el-button>
            <el-button type="danger" size="small" @click="clearLogs">
              清空日志
            </el-button>
          </div>
        </div>
      </template>

      <!-- 日志列表 -->
      <div class="log-container" ref="logContainer">
        <el-timeline>
          <el-timeline-item
            v-for="(log, index) in logs"
            :key="index"
            :type="getLogTypeColor(log.level)"
            :timestamp="log.timestamp"
            placement="top"
          >
            <el-card class="log-item">
              <div class="log-header">
                <el-tag :type="getLogLevelType(log.level)" size="small">
                  {{ log.level }}
                </el-tag>
                <span class="log-source">{{ log.source }}</span>
              </div>
              <div class="log-content" :class="{ 'log-error': log.level === 'ERROR' }">
                {{ log.message }}
              </div>
              <div v-if="log.details" class="log-details">
                <el-collapse>
                  <el-collapse-item title="详细信息">
                    <pre>{{ log.details }}</pre>
                  </el-collapse-item>
                </el-collapse>
              </div>
            </el-card>
          </el-timeline-item>
        </el-timeline>

        <!-- 加载更多 -->
        <div v-if="hasMore" class="load-more">
          <el-button type="text" :loading="loading" @click="loadMore">
            加载更多
          </el-button>
        </div>
      </div>
    </el-card>

    <!-- 实时日志统计 -->
    <el-row :gutter="20" class="stat-row">
      <el-col :span="6" v-for="(stat, index) in logStats" :key="index">
        <el-card class="stat-card">
          <div class="stat-content">
            <div class="stat-value" :style="{ color: stat.color }">
              {{ stat.value }}
            </div>
            <div class="stat-label">{{ stat.label }}</div>
          </div>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted, nextTick } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import axios from 'axios'

// 过滤表单
const filterForm = ref({
  level: [],
  dateRange: [],
  keyword: ''
})

// 日志数据
const logs = ref([])
const loading = ref(false)
const hasMore = ref(true)
const autoScroll = ref(true)
const logContainer = ref(null)

// 日志统计
const logStats = ref([
  { label: '总日志数', value: '0', color: '#409EFF' },
  { label: '错误数', value: '0', color: '#F56C6C' },
  { label: '警告数', value: '0', color: '#E6A23C' },
  { label: '今日日志', value: '0', color: '#67C23A' }
])

// WebSocket连接
let ws = null

// 初始化WebSocket
const initWebSocket = () => {
  ws = new WebSocket('ws://localhost:8000/ws/logs')
  
  ws.onmessage = (event) => {
    const log = JSON.parse(event.data)
    logs.value.unshift(log)
    updateLogStats()
    
    if (autoScroll.value && logContainer.value) {
      nextTick(() => {
        logContainer.value.scrollTop = 0
      })
    }
  }
  
  ws.onclose = () => {
    console.log('WebSocket连接已关闭')
    // 尝试重新连接
    setTimeout(initWebSocket, 5000)
  }
  
  ws.onerror = (error) => {
    console.error('WebSocket错误:', error)
  }
}

// 获取历史日志
const fetchLogs = async () => {
  loading.value = true
  try {
    const response = await axios.get('/api/logs', {
      params: {
        ...filterForm.value,
        startTime: filterForm.value.dateRange?.[0],
        endTime: filterForm.value.dateRange?.[1]
      }
    })
    logs.value = response.data.logs
    hasMore.value = response.data.hasMore
    updateLogStats()
  } catch (error) {
    ElMessage.error('获取日志失败')
  } finally {
    loading.value = false
  }
}

// 加载更多日志
const loadMore = async () => {
  if (loading.value) return
  
  loading.value = true
  try {
    const lastLog = logs.value[logs.value.length - 1]
    const response = await axios.get('/api/logs', {
      params: {
        ...filterForm.value,
        before: lastLog.timestamp
      }
    })
    logs.value.push(...response.data.logs)
    hasMore.value = response.data.hasMore
  } catch (error) {
    ElMessage.error('加载更多日志失败')
  } finally {
    loading.value = false
  }
}

// 更新日志统计
const updateLogStats = () => {
  const today = new Date().toISOString().split('T')[0]
  const stats = {
    total: logs.value.length,
    error: logs.value.filter(log => log.level === 'ERROR').length,
    warning: logs.value.filter(log => log.level === 'WARNING').length,
    today: logs.value.filter(log => log.timestamp.startsWith(today)).length
  }
  
  logStats.value[0].value = stats.total.toString()
  logStats.value[1].value = stats.error.toString()
  logStats.value[2].value = stats.warning.toString()
  logStats.value[3].value = stats.today.toString()
}

// 导出日志
const exportLogs = async () => {
  try {
    const response = await axios.get('/api/logs/export', {
      params: filterForm.value,
      responseType: 'blob'
    })
    const url = window.URL.createObjectURL(new Blob([response.data]))
    const link = document.createElement('a')
    link.href = url
    link.download = `system_logs_${new Date().toISOString()}.txt`
    link.click()
    window.URL.revokeObjectURL(url)
  } catch (error) {
    ElMessage.error('导出日志失败')
  }
}

// 清空日志
const clearLogs = async () => {
  try {
    await ElMessageBox.confirm('确定要清空所有日志吗？此操作不可恢复！', '警告', {
      type: 'warning',
      confirmButtonText: '确定清空',
      confirmButtonClass: 'el-button--danger'
    })
    
    await axios.delete('/api/logs')
    logs.value = []
    updateLogStats()
    ElMessage.success('日志已清空')
  } catch (error) {
    if (error !== 'cancel') {
      ElMessage.error('清空日志失败')
    }
  }
}

// 搜索处理
const handleSearch = () => {
  fetchLogs()
}

// 重置表单
const resetForm = () => {
  filterForm.value = {
    level: [],
    dateRange: [],
    keyword: ''
  }
  handleSearch()
}

// 工具函数
const getLogTypeColor = (level: string) => {
  const colors = {
    'INFO': '',
    'WARNING': 'warning',
    'ERROR': 'danger',
    'DEBUG': 'info'
  }
  return colors[level] || ''
}

const getLogLevelType = (level: string) => {
  const types = {
    'INFO': '',
    'WARNING': 'warning',
    'ERROR': 'danger',
    'DEBUG': 'info'
  }
  return types[level] || ''
}

// 生命周期钩子
onMounted(() => {
  fetchLogs()
  initWebSocket()
})

onUnmounted(() => {
  if (ws) {
    ws.close()
  }
})
</script>

<style scoped>
.logs {
  padding: 20px;
}

.filter-card {
  margin-bottom: 20px;
}

.filter-form {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
}

.log-card {
  margin-bottom: 20px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.header-left {
  display: flex;
  align-items: center;
}

.header-right {
  display: flex;
  gap: 10px;
}

.log-container {
  height: 600px;
  overflow-y: auto;
}

.log-item {
  margin-bottom: 10px;
}

.log-header {
  display: flex;
  align-items: center;
  margin-bottom: 8px;
}

.log-source {
  margin-left: 10px;
  color: #909399;
  font-size: 12px;
}

.log-content {
  font-family: monospace;
  white-space: pre-wrap;
  word-break: break-all;
}

.log-error {
  color: #F56C6C;
}

.log-details {
  margin-top: 10px;
}

.log-details pre {
  margin: 0;
  padding: 10px;
  background-color: #f8f9fa;
  border-radius: 4px;
  font-size: 12px;
}

.load-more {
  text-align: center;
  margin-top: 20px;
}

.stat-row {
  margin-top: 20px;
}

.stat-card {
  text-align: center;
}

.stat-value {
  font-size: 24px;
  font-weight: bold;
}

.stat-label {
  font-size: 14px;
  color: #909399;
  margin-top: 5px;
}
</style> 