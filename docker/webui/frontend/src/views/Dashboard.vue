<template>
  <div class="dashboard">
    <el-row :gutter="20">
      <!-- 系统状态卡片 -->
      <el-col :span="6" v-for="(stat, index) in systemStats" :key="index">
        <el-card class="stat-card" :body-style="{ padding: '20px' }">
          <div class="stat-content">
            <el-icon :size="24" :class="stat.icon"><component :is="stat.icon" /></el-icon>
            <div class="stat-info">
              <div class="stat-value">{{ stat.value }}</div>
              <div class="stat-label">{{ stat.label }}</div>
            </div>
          </div>
        </el-card>
      </el-col>
    </el-row>

    <el-row :gutter="20" class="chart-row">
      <!-- 分析统计图表 -->
      <el-col :span="12">
        <el-card>
          <template #header>
            <div class="card-header">
              <span>分析统计</span>
              <el-select v-model="timeRange" size="small">
                <el-option label="今日" value="today" />
                <el-option label="本周" value="week" />
                <el-option label="本月" value="month" />
              </el-select>
            </div>
          </template>
          <div ref="analysisChart" class="chart"></div>
        </el-card>
      </el-col>

      <!-- 系统资源监控 -->
      <el-col :span="12">
        <el-card>
          <template #header>
            <div class="card-header">
              <span>资源监控</span>
              <el-switch
                v-model="realTimeMonitoring"
                active-text="实时监控"
              />
            </div>
          </template>
          <div ref="resourceChart" class="chart"></div>
        </el-card>
      </el-col>
    </el-row>

    <el-row :gutter="20">
      <!-- 最近信号列表 -->
      <el-col :span="12">
        <el-card>
          <template #header>
            <div class="card-header">
              <span>最近信号</span>
              <el-button type="text">查看全部</el-button>
            </div>
          </template>
          <el-table :data="recentSignals" style="width: 100%">
            <el-table-column prop="timestamp" label="时间" width="180" />
            <el-table-column prop="symbol" label="交易对" width="120" />
            <el-table-column prop="type" label="类型" width="100" />
            <el-table-column prop="confidence" label="置信度">
              <template #default="scope">
                <el-progress
                  :percentage="scope.row.confidence * 100"
                  :color="getConfidenceColor(scope.row.confidence)"
                />
              </template>
            </el-table-column>
          </el-table>
        </el-card>
      </el-col>

      <!-- 系统日志 -->
      <el-col :span="12">
        <el-card>
          <template #header>
            <div class="card-header">
              <span>系统日志</span>
              <el-button type="text">查看全部</el-button>
            </div>
          </template>
          <div class="log-container">
            <div v-for="(log, index) in systemLogs" :key="index" class="log-item">
              <el-tag
                :type="getLogLevelType(log.level)"
                size="small"
                class="log-level"
              >
                {{ log.level }}
              </el-tag>
              <span class="log-time">{{ log.timestamp }}</span>
              <span class="log-message">{{ log.message }}</span>
            </div>
          </div>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'
import { Monitor, TrendCharts, Connection, Warning } from '@element-plus/icons-vue'
import * as echarts from 'echarts'
import axios from 'axios'

// 系统状态数据
const systemStats = ref([
  { label: 'CPU使用率', value: '0%', icon: 'Monitor' },
  { label: '内存使用率', value: '0%', icon: 'TrendCharts' },
  { label: '磁盘使用率', value: '0%', icon: 'Connection' },
  { label: '系统告警', value: '0', icon: 'Warning' }
])

// 图表相关
const timeRange = ref('today')
const realTimeMonitoring = ref(true)
const analysisChart = ref(null)
const resourceChart = ref(null)
let analysisChartInstance = null
let resourceChartInstance = null

// 最近信号数据
const recentSignals = ref([])
const systemLogs = ref([])

// 获取系统状态
const fetchSystemStatus = async () => {
  try {
    const response = await axios.get('/api/system/status')
    systemStats.value[0].value = `${response.data.cpu.percent}%`
    systemStats.value[1].value = `${response.data.memory.percent}%`
    systemStats.value[2].value = `${response.data.disk.percent}%`
  } catch (error) {
    console.error('获取系统状态失败:', error)
  }
}

// 获取分析统计
const fetchAnalysisStats = async () => {
  try {
    const response = await axios.get('/api/analysis/stats')
    recentSignals.value = response.data.recent_signals
  } catch (error) {
    console.error('获取分析统计失败:', error)
  }
}

// 获取系统日志
const fetchSystemLogs = async () => {
  try {
    const response = await axios.get('/api/logs')
    systemLogs.value = response.data.logs
  } catch (error) {
    console.error('获取系统日志失败:', error)
  }
}

// 初始化分析图表
const initAnalysisChart = () => {
  if (analysisChart.value) {
    analysisChartInstance = echarts.init(analysisChart.value)
    const option = {
      title: {
        text: '分析结果统计'
      },
      tooltip: {
        trigger: 'axis'
      },
      xAxis: {
        type: 'category',
        data: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
      },
      yAxis: {
        type: 'value'
      },
      series: [{
        data: [150, 230, 224, 218, 135, 147, 260],
        type: 'line'
      }]
    }
    analysisChartInstance.setOption(option)
  }
}

// 初始化资源监控图表
const initResourceChart = () => {
  if (resourceChart.value) {
    resourceChartInstance = echarts.init(resourceChart.value)
    const option = {
      title: {
        text: '资源使用趋势'
      },
      tooltip: {
        trigger: 'axis'
      },
      legend: {
        data: ['CPU', '内存', '磁盘']
      },
      xAxis: {
        type: 'time',
        splitLine: {
          show: false
        }
      },
      yAxis: {
        type: 'value',
        max: 100,
        splitLine: {
          show: true
        }
      },
      series: [
        {
          name: 'CPU',
          type: 'line',
          data: []
        },
        {
          name: '内存',
          type: 'line',
          data: []
        },
        {
          name: '磁盘',
          type: 'line',
          data: []
        }
      ]
    }
    resourceChartInstance.setOption(option)
  }
}

// 获取置信度颜色
const getConfidenceColor = (confidence: number) => {
  if (confidence >= 0.8) return '#67C23A'
  if (confidence >= 0.6) return '#E6A23C'
  return '#F56C6C'
}

// 获取日志级别样式
const getLogLevelType = (level: string) => {
  const types = {
    'INFO': '',
    'WARNING': 'warning',
    'ERROR': 'danger',
    'DEBUG': 'info'
  }
  return types[level] || ''
}

// 组件挂载时初始化
onMounted(() => {
  fetchSystemStatus()
  fetchAnalysisStats()
  fetchSystemLogs()
  initAnalysisChart()
  initResourceChart()
  
  // 设置定时更新
  const statusInterval = setInterval(fetchSystemStatus, 30000)
  const statsInterval = setInterval(fetchAnalysisStats, 60000)
  const logsInterval = setInterval(fetchSystemLogs, 10000)
  
  onUnmounted(() => {
    clearInterval(statusInterval)
    clearInterval(statsInterval)
    clearInterval(logsInterval)
    if (analysisChartInstance) analysisChartInstance.dispose()
    if (resourceChartInstance) resourceChartInstance.dispose()
  })
})
</script>

<style scoped>
.dashboard {
  padding: 20px;
}

.stat-card {
  margin-bottom: 20px;
}

.stat-content {
  display: flex;
  align-items: center;
}

.stat-info {
  margin-left: 20px;
}

.stat-value {
  font-size: 24px;
  font-weight: bold;
  color: #303133;
}

.stat-label {
  font-size: 14px;
  color: #909399;
  margin-top: 5px;
}

.chart-row {
  margin: 20px 0;
}

.chart {
  height: 300px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.log-container {
  height: 300px;
  overflow-y: auto;
}

.log-item {
  padding: 8px 0;
  border-bottom: 1px solid #EBEEF5;
}

.log-level {
  margin-right: 10px;
}

.log-time {
  color: #909399;
  margin-right: 10px;
  font-size: 12px;
}

.log-message {
  color: #606266;
}
</style> 