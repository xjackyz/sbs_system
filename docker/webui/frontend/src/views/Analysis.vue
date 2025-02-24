<template>
  <div class="analysis">
    <!-- 搜索和过滤区域 -->
    <el-card class="filter-card">
      <el-form :inline="true" :model="filterForm" class="filter-form">
        <el-form-item label="交易对">
          <el-select v-model="filterForm.symbol" placeholder="选择交易对">
            <el-option label="NQ1!" value="NQ1!" />
            <el-option label="ES1!" value="ES1!" />
            <el-option label="YM1!" value="YM1!" />
          </el-select>
        </el-form-item>
        <el-form-item label="信号类型">
          <el-select v-model="filterForm.signalType" placeholder="选择信号类型">
            <el-option label="做多" value="long" />
            <el-option label="做空" value="short" />
          </el-select>
        </el-form-item>
        <el-form-item label="时间范围">
          <el-date-picker
            v-model="filterForm.dateRange"
            type="daterange"
            range-separator="至"
            start-placeholder="开始日期"
            end-placeholder="结束日期"
          />
        </el-form-item>
        <el-form-item label="置信度">
          <el-slider
            v-model="filterForm.confidence"
            range
            :marks="{
              0: '0%',
              50: '50%',
              80: '80%',
              100: '100%'
            }"
          />
        </el-form-item>
        <el-form-item>
          <el-button type="primary" @click="handleSearch">搜索</el-button>
          <el-button @click="resetForm">重置</el-button>
        </el-form-item>
      </el-form>
    </el-card>

    <!-- 分析结果统计 -->
    <el-row :gutter="20" class="stat-row">
      <el-col :span="6" v-for="(stat, index) in analysisStats" :key="index">
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

    <!-- 分析记录表格 -->
    <el-card class="table-card">
      <template #header>
        <div class="card-header">
          <span>分析记录</span>
          <div class="header-actions">
            <el-button type="primary" size="small" @click="exportData">
              导出数据
            </el-button>
            <el-button type="success" size="small" @click="refreshData">
              刷新
            </el-button>
          </div>
        </div>
      </template>
      
      <el-table
        :data="analysisRecords"
        style="width: 100%"
        v-loading="loading"
        @row-click="handleRowClick"
      >
        <el-table-column prop="timestamp" label="时间" width="180" sortable />
        <el-table-column prop="symbol" label="交易对" width="100" />
        <el-table-column prop="signalType" label="信号类型" width="100">
          <template #default="scope">
            <el-tag :type="scope.row.signalType === 'long' ? 'success' : 'danger'">
              {{ scope.row.signalType === 'long' ? '做多' : '做空' }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="confidence" label="置信度" width="150">
          <template #default="scope">
            <el-progress
              :percentage="scope.row.confidence * 100"
              :color="getConfidenceColor(scope.row.confidence)"
            />
          </template>
        </el-table-column>
        <el-table-column prop="entryPrice" label="入场价格" width="120" />
        <el-table-column prop="stopLoss" label="止损价格" width="120" />
        <el-table-column prop="takeProfit" label="目标价格" width="120" />
        <el-table-column prop="status" label="状态" width="100">
          <template #default="scope">
            <el-tag :type="getStatusType(scope.row.status)">
              {{ scope.row.status }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column label="操作" fixed="right" width="150">
          <template #default="scope">
            <el-button
              size="small"
              @click.stop="viewDetails(scope.row)"
            >
              详情
            </el-button>
            <el-button
              size="small"
              type="danger"
              @click.stop="deleteRecord(scope.row)"
            >
              删除
            </el-button>
          </template>
        </el-table-column>
      </el-table>

      <div class="pagination-container">
        <el-pagination
          v-model:current-page="currentPage"
          v-model:page-size="pageSize"
          :page-sizes="[10, 20, 50, 100]"
          :total="total"
          layout="total, sizes, prev, pager, next, jumper"
          @size-change="handleSizeChange"
          @current-change="handleCurrentChange"
        />
      </div>
    </el-card>

    <!-- 详情对话框 -->
    <el-dialog
      v-model="detailsVisible"
      title="分析详情"
      width="80%"
      destroy-on-close
    >
      <el-descriptions :column="2" border>
        <el-descriptions-item label="交易对">
          {{ currentRecord.symbol }}
        </el-descriptions-item>
        <el-descriptions-item label="信号类型">
          {{ currentRecord.signalType === 'long' ? '做多' : '做空' }}
        </el-descriptions-item>
        <el-descriptions-item label="入场价格">
          {{ currentRecord.entryPrice }}
        </el-descriptions-item>
        <el-descriptions-item label="止损价格">
          {{ currentRecord.stopLoss }}
        </el-descriptions-item>
        <el-descriptions-item label="目标价格">
          {{ currentRecord.takeProfit }}
        </el-descriptions-item>
        <el-descriptions-item label="置信度">
          {{ (currentRecord.confidence * 100).toFixed(2) }}%
        </el-descriptions-item>
      </el-descriptions>

      <!-- 图表标注 -->
      <div class="chart-container">
        <img
          v-if="currentRecord.chartImage"
          :src="currentRecord.chartImage"
          class="chart-image"
        />
      </div>

      <!-- 分析结果 -->
      <el-collapse v-model="activeCollapse">
        <el-collapse-item title="序列评估" name="sequence">
          <el-descriptions :column="1" border>
            <el-descriptions-item label="有效性">
              {{ currentRecord.sequenceEvaluation?.validity }}
            </el-descriptions-item>
            <el-descriptions-item label="完整度">
              {{ currentRecord.sequenceEvaluation?.completeness }}%
            </el-descriptions-item>
            <el-descriptions-item label="可信度">
              {{ currentRecord.sequenceEvaluation?.confidence }}%
            </el-descriptions-item>
          </el-descriptions>
        </el-collapse-item>

        <el-collapse-item title="关键点位" name="points">
          <el-descriptions :column="2" border>
            <el-descriptions-item label="突破点">
              {{ currentRecord.keyPoints?.breakout }}
            </el-descriptions-item>
            <el-descriptions-item label="Point 1">
              {{ currentRecord.keyPoints?.point1 }}
            </el-descriptions-item>
            <el-descriptions-item label="Point 2">
              {{ currentRecord.keyPoints?.point2 }}
            </el-descriptions-item>
            <el-descriptions-item label="Point 3">
              {{ currentRecord.keyPoints?.point3 }}
            </el-descriptions-item>
            <el-descriptions-item label="Point 4">
              {{ currentRecord.keyPoints?.point4 }}
            </el-descriptions-item>
          </el-descriptions>
        </el-collapse-item>

        <el-collapse-item title="趋势分析" name="trend">
          <el-descriptions :column="1" border>
            <el-descriptions-item label="SMA20趋势">
              {{ currentRecord.trendAnalysis?.sma20 }}
            </el-descriptions-item>
            <el-descriptions-item label="SMA200趋势">
              {{ currentRecord.trendAnalysis?.sma200 }}
            </el-descriptions-item>
            <el-descriptions-item label="整体趋势">
              {{ currentRecord.trendAnalysis?.overall }}
            </el-descriptions-item>
          </el-descriptions>
        </el-collapse-item>

        <el-collapse-item title="风险评估" name="risk">
          <el-descriptions :column="1" border>
            <el-descriptions-item label="风险等级">
              <el-tag :type="getRiskLevelType(currentRecord.riskAssessment?.level)">
                {{ currentRecord.riskAssessment?.level }}
              </el-tag>
            </el-descriptions-item>
            <el-descriptions-item label="主要风险点">
              {{ currentRecord.riskAssessment?.mainRisks }}
            </el-descriptions-item>
          </el-descriptions>
        </el-collapse-item>
      </el-collapse>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import axios from 'axios'

// 过滤表单
const filterForm = ref({
  symbol: '',
  signalType: '',
  dateRange: [],
  confidence: [0, 100]
})

// 分析统计数据
const analysisStats = ref([
  { label: '总分析次数', value: '0', color: '#409EFF' },
  { label: '成功信号', value: '0', color: '#67C23A' },
  { label: '平均置信度', value: '0%', color: '#E6A23C' },
  { label: '成功率', value: '0%', color: '#F56C6C' }
])

// 表格数据
const loading = ref(false)
const currentPage = ref(1)
const pageSize = ref(20)
const total = ref(0)
const analysisRecords = ref([])

// 详情对话框
const detailsVisible = ref(false)
const currentRecord = ref({})
const activeCollapse = ref(['sequence'])

// 获取分析记录
const fetchAnalysisRecords = async () => {
  loading.value = true
  try {
    const response = await axios.get('/api/analysis/records', {
      params: {
        page: currentPage.value,
        pageSize: pageSize.value,
        ...filterForm.value
      }
    })
    analysisRecords.value = response.data.records
    total.value = response.data.total
  } catch (error) {
    ElMessage.error('获取分析记录失败')
  } finally {
    loading.value = false
  }
}

// 搜索处理
const handleSearch = () => {
  currentPage.value = 1
  fetchAnalysisRecords()
}

// 重置表单
const resetForm = () => {
  filterForm.value = {
    symbol: '',
    signalType: '',
    dateRange: [],
    confidence: [0, 100]
  }
  handleSearch()
}

// 分页处理
const handleSizeChange = (val: number) => {
  pageSize.value = val
  fetchAnalysisRecords()
}

const handleCurrentChange = (val: number) => {
  currentPage.value = val
  fetchAnalysisRecords()
}

// 查看详情
const viewDetails = (row: any) => {
  currentRecord.value = row
  detailsVisible.value = true
}

// 删除记录
const deleteRecord = async (row: any) => {
  try {
    await ElMessageBox.confirm('确定要删除这条记录吗？', '提示', {
      type: 'warning'
    })
    await axios.delete(`/api/analysis/records/${row.id}`)
    ElMessage.success('删除成功')
    fetchAnalysisRecords()
  } catch (error) {
    if (error !== 'cancel') {
      ElMessage.error('删除失败')
    }
  }
}

// 导出数据
const exportData = async () => {
  try {
    const response = await axios.get('/api/analysis/export', {
      params: filterForm.value,
      responseType: 'blob'
    })
    const url = window.URL.createObjectURL(new Blob([response.data]))
    const link = document.createElement('a')
    link.href = url
    link.download = `analysis_records_${new Date().toISOString()}.xlsx`
    link.click()
    window.URL.revokeObjectURL(url)
  } catch (error) {
    ElMessage.error('导出失败')
  }
}

// 刷新数据
const refreshData = () => {
  fetchAnalysisRecords()
}

// 工具函数
const getConfidenceColor = (confidence: number) => {
  if (confidence >= 0.8) return '#67C23A'
  if (confidence >= 0.6) return '#E6A23C'
  return '#F56C6C'
}

const getStatusType = (status: string) => {
  const types = {
    '进行中': 'warning',
    '已完成': 'success',
    '已取消': 'info',
    '已失败': 'danger'
  }
  return types[status] || ''
}

const getRiskLevelType = (level: string) => {
  const types = {
    '低': 'success',
    '中': 'warning',
    '高': 'danger'
  }
  return types[level] || ''
}

onMounted(() => {
  fetchAnalysisRecords()
})
</script>

<style scoped>
.analysis {
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

.stat-row {
  margin-bottom: 20px;
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

.table-card {
  margin-bottom: 20px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.header-actions {
  display: flex;
  gap: 10px;
}

.pagination-container {
  margin-top: 20px;
  display: flex;
  justify-content: flex-end;
}

.chart-container {
  margin: 20px 0;
  text-align: center;
}

.chart-image {
  max-width: 100%;
  max-height: 500px;
  object-fit: contain;
}
</style> 