<template>
  <el-container class="app-container">
    <el-aside width="200px">
      <el-menu
        default-active="dashboard"
        class="el-menu-vertical"
        @select="handleSelect"
        background-color="#545c64"
        text-color="#fff"
        active-text-color="#ffd04b">
        <el-menu-item index="dashboard">
          <el-icon><DataLine /></el-icon>
          <span>仪表盘</span>
        </el-menu-item>
        <el-menu-item index="analysis">
          <el-icon><TrendCharts /></el-icon>
          <span>分析记录</span>
        </el-menu-item>
        <el-menu-item index="logs">
          <el-icon><Document /></el-icon>
          <span>系统日志</span>
        </el-menu-item>
        <el-menu-item index="settings">
          <el-icon><Setting /></el-icon>
          <span>系统设置</span>
        </el-menu-item>
      </el-menu>
    </el-aside>
    
    <el-container>
      <el-header>
        <div class="header-content">
          <h2>SBS Trading System</h2>
          <el-dropdown>
            <el-button type="primary">
              系统状态
              <el-icon class="el-icon--right"><arrow-down /></el-icon>
            </el-button>
            <template #dropdown>
              <el-dropdown-menu>
                <el-dropdown-item>CPU: {{ systemStatus.cpu }}%</el-dropdown-item>
                <el-dropdown-item>内存: {{ systemStatus.memory }}%</el-dropdown-item>
                <el-dropdown-item>磁盘: {{ systemStatus.disk }}%</el-dropdown-item>
              </el-dropdown-menu>
            </template>
          </el-dropdown>
        </div>
      </el-header>
      
      <el-main>
        <router-view></router-view>
      </el-main>
    </el-container>
  </el-container>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { DataLine, TrendCharts, Document, Setting } from '@element-plus/icons-vue'
import axios from 'axios'

const router = useRouter()
const systemStatus = ref({
  cpu: 0,
  memory: 0,
  disk: 0
})

const handleSelect = (key: string) => {
  router.push(`/${key}`)
}

const fetchSystemStatus = async () => {
  try {
    const response = await axios.get('/api/system/status')
    systemStatus.value = {
      cpu: response.data.cpu.percent,
      memory: response.data.memory.percent,
      disk: response.data.disk.percent
    }
  } catch (error) {
    console.error('获取系统状态失败:', error)
  }
}

onMounted(() => {
  fetchSystemStatus()
  // 每30秒更新一次系统状态
  setInterval(fetchSystemStatus, 30000)
})
</script>

<style scoped>
.app-container {
  height: 100vh;
}

.el-aside {
  background-color: #545c64;
}

.el-menu-vertical {
  height: 100%;
  border-right: none;
}

.el-header {
  background-color: #fff;
  border-bottom: 1px solid #dcdfe6;
  padding: 0 20px;
}

.header-content {
  display: flex;
  justify-content: space-between;
  align-items: center;
  height: 100%;
}

.el-main {
  background-color: #f5f7fa;
  padding: 20px;
}
</style> 