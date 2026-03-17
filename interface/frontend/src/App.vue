<script setup>
import { ref, watch, computed } from 'vue'
import Button from './components/Button.vue'
import AlertsList from './components/AlertsList.vue'
import SystemStatus from './components/SystemStatus.vue'
import { useAlerts } from '@/composables/useAlerts'
import { useApi } from '@/composables/api'

const { alerts, status, acknowledgeAlert } = useAlerts()
const { getFaultHistory, getRawAlerts } = useApi()

const currentSection = ref('Overview')
const faultHistory = ref([])
const loadingFaultHistory = ref(false)
const faultHistoryError = ref(null)

// Raw alerts for export
const rawAlerts = ref([])
const loadingRawAlerts = ref(false)
const rawAlertsError = ref(null)

// Filter state for Fault History
const selectedFanId = ref('')
const selectedFaultType = ref('')

// Filter state for Export
const exportFanId = ref('')
const exportFaultType = ref('')
const exportMode = ref('fault_periods') // fault_periods | raw_alerts

const sections = [
  'Overview',
  'Fault History',
  'Export',
]

const setSection = (section) => {
  currentSection.value = section
}

// Get unique values for filter dropdowns
const uniqueFanIds = computed(() => {
  const assetIds = [...new Set(faultHistory.value.map(f => f.asset_id).filter(Boolean))]
  return assetIds.sort()
})

const uniqueFaultTypes = computed(() => {
  const faultTypes = [...new Set(faultHistory.value.map(f => f.fault_type))]
  return faultTypes.sort()
})

// Filtered fault history
const filteredFaultHistory = computed(() => {
  return faultHistory.value.filter(fault => {
    const matchesFanId = !selectedFanId.value || fault.asset_id === selectedFanId.value
    const matchesFaultType = !selectedFaultType.value || fault.fault_type === selectedFaultType.value
    return matchesFanId && matchesFaultType
  })
})

// Clear all filters
const clearFilters = () => {
  selectedFanId.value = ''
  selectedFaultType.value = ''
}

// Format timestamp for display
const formatTimestamp = (timestamp) => {
  if (!timestamp) return 'N/A'
  // support numbers (epoch seconds/ms) and ISO strings
  if (typeof timestamp === 'string') return new Date(timestamp).toLocaleString()
  const ms = timestamp < 1e12 ? timestamp * 1000 : timestamp
  return new Date(ms).toLocaleString()
}

// Data to export based on filters
const exportData = computed(() => {
  return faultHistory.value.filter(fault => {
    const matchesFanId = !exportFanId.value || fault.asset_id === exportFanId.value
    const matchesFaultType = !exportFaultType.value || fault.fault_type === exportFaultType.value
    return matchesFanId && matchesFaultType
  })
})

const uniqueRawAssetIds = computed(() => {
  const ids = [...new Set(rawAlerts.value.map(a => a.asset_id).filter(Boolean))]
  return ids.sort()
})

const uniqueRawConditions = computed(() => {
  const conds = [...new Set(rawAlerts.value.map(a => a.condition_name || a.message).filter(Boolean))]
  return conds.sort()
})

const exportRawData = computed(() => {
  return rawAlerts.value.filter(a => {
    const matchesFanId = !exportFanId.value || a.asset_id === exportFanId.value
    const cond = a.condition_name || a.message || ''
    const matchesFaultType = !exportFaultType.value || cond === exportFaultType.value
    return matchesFanId && matchesFaultType
  })
})

// Convert data to CSV format
const convertFaultPeriodsToCSV = (data) => {
  if (data.length === 0) return ''
  
  // CSV headers
  const headers = ['Asset ID', 'Fault Type', 'Start Time', 'End Time', 'Acknowledged', 'Acknowledged At']
  
  // CSV rows
  const rows = data.map(fault => {
    const start = fault.start_ts || ''
    const end = fault.end_ts || ''
    return [
      fault.asset_id,
      fault.fault_type,
      start,
      end,
      fault.acknowledged ? 'true' : 'false',
      fault.acknowledged_at || ''
    ]
  })
  
  // Combine headers and rows
  const csvContent = [
    headers.join(','),
    ...rows.map(row => row.map(cell => `"${cell}"`).join(','))
  ].join('\n')
  
  return csvContent
}

const convertRawAlertsToCSV = (data) => {
  if (data.length === 0) return ''

  const headers = [
    'Alert ID',
    'Asset ID',
    'Condition ID',
    'Condition Name',
    'Message',
    'Confidence',
    'Model Timestamp',
    'Inserted At',
  ]

  const rows = data.map(a => ([
    a._id || '',
    a.asset_id || '',
    a.condition_id ?? '',
    a.condition_name || '',
    a.message || '',
    a.confidence ?? '',
    a.ts ?? '',
    a.timestamp || '',
  ]))

  return [
    headers.join(','),
    ...rows.map(row => row.map(cell => `"${String(cell).replace(/"/g, '""')}"`).join(',')),
  ].join('\n')
}

// Download CSV file
const downloadCSV = () => {
  const rows = exportMode.value === 'raw_alerts' ? exportRawData.value : exportData.value
  if (rows.length === 0) {
    alert('No data to export. Please adjust your filters or ensure data is loaded.')
    return
  }
  
  const csvContent =
    exportMode.value === 'raw_alerts'
      ? convertRawAlertsToCSV(rows)
      : convertFaultPeriodsToCSV(rows)
  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' })
  const link = document.createElement('a')
  const url = URL.createObjectURL(blob)
  
  // Generate filename with filters
  let filename = exportMode.value === 'raw_alerts' ? 'raw_alerts' : 'fault_periods'
  if (exportFanId.value) filename += `_${exportFanId.value}`
  if (exportFaultType.value) filename += `_${exportFaultType.value.replace(/\s+/g, '_').toLowerCase()}`
  filename += '.csv'
  
  link.setAttribute('href', url)
  link.setAttribute('download', filename)
  link.style.visibility = 'hidden'
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
}

// Clear export filters
const clearExportFilters = () => {
  exportFanId.value = ''
  exportFaultType.value = ''
}

// Fetch fault history when switching to Fault History or Export tab
watch(currentSection, async (newSection) => {
  if (newSection === 'Fault History' || newSection === 'Export') {
    // Only fetch if we don't have data yet
    if (faultHistory.value.length === 0 && !loadingFaultHistory.value) {
      loadingFaultHistory.value = true
      faultHistoryError.value = null
      try {
        faultHistory.value = await getFaultHistory()
      } catch (error) {
        faultHistoryError.value = 'Failed to load fault history'
        console.error('Error loading fault history:', error)
      } finally {
        loadingFaultHistory.value = false
      }
    }
  }
})

watch(currentSection, async (newSection) => {
  if (newSection !== 'Export') return
  if (rawAlerts.value.length > 0 || loadingRawAlerts.value) return

  loadingRawAlerts.value = true
  rawAlertsError.value = null
  try {
    rawAlerts.value = await getRawAlerts()
  } catch (error) {
    rawAlertsError.value = 'Failed to load raw alerts'
    console.error('Error loading raw alerts:', error)
  } finally {
    loadingRawAlerts.value = false
  }
})
</script>

<template>
  <div class="container">
    <div class="nav-bar">
      <h2 id="company-logo">Delta Controls</h2>
      <div class="nav-bar-content">
        <Button 
          v-for="section in sections" 
          :key="section"
          :label="section"
          :is-active="currentSection === section"
          @click="setSection(section)"
        />
      </div>
    </div>
    <div class="main-content" >
      <div class="main-content-header">
        <h2>{{ currentSection }}</h2>
      </div>
      <div class="section-context">
        <!-- Overview Section -->
        <template v-if="currentSection === 'Overview'">
          <AlertsList :alerts="alerts" :acknowledge-alert="acknowledgeAlert" class="alerts-list"/>
          <SystemStatus :alerts="alerts" class="system-status"/>
        </template>
        
        <!-- Asset Management Section -->
        <template v-else-if="currentSection === 'Asset Management'">
          <div class="section-placeholder">
            <h3>Asset Management</h3>
            <p>Asset management content will be displayed here.</p>
          </div>
        </template>
        
        <!-- Fault History Section -->
        <template v-else-if="currentSection === 'Fault History'">
          <div class="fault-history-section">
            <div v-if="loadingFaultHistory" class="loading">
              <p>Loading fault history...</p>
            </div>
            <div v-else-if="faultHistoryError" class="error">
              <p>{{ faultHistoryError }}</p>
            </div>
            <div v-else-if="faultHistory.length === 0" class="no-data">
              <p>No fault history found.</p>
            </div>
            <div v-else>
              <!-- Filter Controls -->
              <div class="filters-container">
                <div class="filter-group">
                  <label for="fan-id-filter">Filter by Fan ID:</label>
                  <select 
                    id="fan-id-filter" 
                    v-model="selectedFanId" 
                    class="filter-select"
                  >
                    <option value="">All Fans</option>
                    <option 
                      v-for="fanId in uniqueFanIds" 
                      :key="fanId" 
                      :value="fanId"
                    >
                      Fan {{ fanId }}
                    </option>
                  </select>
                </div>
                <div class="filter-group">
                  <label for="fault-type-filter">Filter by Fault Type:</label>
                  <select 
                    id="fault-type-filter" 
                    v-model="selectedFaultType" 
                    class="filter-select"
                  >
                    <option value="">All Fault Types</option>
                    <option 
                      v-for="faultType in uniqueFaultTypes" 
                      :key="faultType" 
                      :value="faultType"
                    >
                      {{ faultType }}
                    </option>
                  </select>
                </div>
                <button 
                  @click="clearFilters" 
                  class="clear-filters-btn"
                  :disabled="!selectedFanId && !selectedFaultType"
                >
                  Clear Filters
                </button>
              </div>
              
              <!-- Results Count -->
              <div class="results-info">
                <p>
                  Showing {{ filteredFaultHistory.length }} of {{ faultHistory.length }} records
                </p>
              </div>

              <!-- Fault History Table -->
              <div v-if="filteredFaultHistory.length === 0" class="no-data">
                <p>No records match the selected filters.</p>
              </div>
              <div v-else class="fault-history-table">
                <table>
                  <thead>
                    <tr>
                      <th>Asset ID</th>
                      <th>Fault Type</th>
                      <th>Start Time</th>
                      <th>End Time</th>
                      <th>Acknowledged</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr v-for="(fault, index) in filteredFaultHistory" :key="index">
                      <td>{{ fault.asset_id }}</td>
                      <td>{{ fault.fault_type }}</td>
                      <td>{{ formatTimestamp(fault.start_ts) }}</td>
                      <td>{{ formatTimestamp(fault.end_ts) }}</td>
                      <td>{{ fault.acknowledged ? 'Yes' : 'No' }}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </template>
        
        <!-- Export Section -->
        <template v-else-if="currentSection === 'Export'">
          <div class="export-section">
            <div v-if="loadingFaultHistory || loadingRawAlerts" class="loading">
              <p>Loading data...</p>
            </div>
            <div v-else-if="faultHistoryError || rawAlertsError" class="error">
              <p>{{ faultHistoryError || rawAlertsError }}</p>
            </div>
            <div v-else-if="faultHistory.length === 0 && rawAlerts.length === 0" class="no-data">
              <p>No data available to export.</p>
            </div>
            <div v-else>
              <div class="export-header">
                <h3>Export Data</h3>
                <p class="export-description">
                  Choose what to export, then optionally filter.
                </p>
              </div>

              <div class="filters-container">
                <div class="filter-group">
                  <label for="export-mode">Export type:</label>
                  <select id="export-mode" v-model="exportMode" class="filter-select">
                    <option value="fault_periods">Fault periods</option>
                    <option value="raw_alerts">Raw alerts (confidence, etc.)</option>
                  </select>
                </div>
              </div>
              
              <!-- Export Filter Controls -->
              <div class="filters-container">
                <div class="filter-group">
                  <label for="export-fan-id-filter">Filter by Asset ID:</label>
                  <select 
                    id="export-fan-id-filter" 
                    v-model="exportFanId" 
                    class="filter-select"
                  >
                    <option value="">All Fans</option>
                    <option 
                      v-for="assetId in (exportMode === 'raw_alerts' ? uniqueRawAssetIds : uniqueFanIds)" 
                      :key="assetId" 
                      :value="assetId"
                    >
                      {{ assetId }}
                    </option>
                  </select>
                </div>
                <div class="filter-group">
                  <label for="export-fault-type-filter">Filter by Fault Type:</label>
                  <select 
                    id="export-fault-type-filter" 
                    v-model="exportFaultType" 
                    class="filter-select"
                  >
                    <option value="">All Fault Types</option>
                    <option 
                      v-for="faultType in (exportMode === 'raw_alerts' ? uniqueRawConditions : uniqueFaultTypes)" 
                      :key="faultType" 
                      :value="faultType"
                    >
                      {{ faultType }}
                    </option>
                  </select>
                </div>
                <button 
                  @click="clearExportFilters" 
                  class="clear-filters-btn"
                  :disabled="!exportFanId && !exportFaultType"
                >
                  Clear Filters
                </button>
              </div>
              
              <!-- Export Info -->
              <div class="export-info">
                <div class="export-stats">
                  <p v-if="exportMode === 'fault_periods'"><strong>Records to export:</strong> {{ exportData.length }} of {{ faultHistory.length }}</p>
                  <p v-else><strong>Records to export:</strong> {{ exportRawData.length }} of {{ rawAlerts.length }}</p>
                  <p v-if="exportFanId" class="filter-info">• Filtered by Fan ID: {{ exportFanId }}</p>
                  <p v-if="exportFaultType" class="filter-info">• Filtered by Fault Type: {{ exportFaultType }}</p>
                </div>
              </div>
              
              <!-- Download Button -->
              <div class="export-actions">
                <button 
                  @click="downloadCSV" 
                  class="export-btn"
                  :disabled="(exportMode === 'fault_periods' ? exportData.length : exportRawData.length) === 0"
                >
                  <span class="export-icon">📥</span>
                  Download CSV
                </button>
                <p class="export-hint">
                  The file will be saved as a CSV file with all selected data.
                </p>
              </div>
            </div>
          </div>
        </template>
        
      </div>
    </div>
  </div>
</template>

<style scoped>
.container {
  display: flex;
  flex-direction: row;
  width: 100%;
  height: 100vh;
  background-color:#F2F2F2  ;
  margin: 0;
  padding: 0;
}
 /* Nav Bar  styles */
.nav-bar {
  width: 200px;
  background-color: #FFFFFF;
  padding: 20px;
}
#company-logo {
  font-size: 24px;
  font-weight: bold;
  text-align: center;
  border: 2px solid #FFFFFF;
}

.nav-bar-content {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

 /* Main Content styles */
.main-content {
  flex: 1;
  display: flex;
  flex-direction: column;
  background-color: #FFFFFF;
  margin: 20px;
  padding: 20px;
  border: 2px solid #0087DC;
  border-radius: 10px;
}

.main-content-header {
  padding: 16px 0;
  height: 5%;
  border-bottom: 2px solid #0087DC;
  margin-bottom: 20px;
}

.main-content-header h2 {
  margin: 0;
  color: #333;
  font-size: 28px;
}

.section-context {
  flex: 1;
  height: 95%;
}

.section-placeholder {
  padding: 40px;
  text-align: center;
  color: #666;
}

.section-placeholder h3 {
  margin: 0 0 16px 0;
  color: #333;
  font-size: 24px;
}

.section-placeholder p {
  margin: 0;
  font-size: 16px;
}

/* Fault History Section */
.fault-history-section {
  padding: 20px;
}

.loading, .error, .no-data {
  text-align: center;
  padding: 40px;
  color: #666;
}

.error {
  color: #d32f2f;
}

/* Filter Controls */
.filters-container {
  display: flex;
  gap: 20px;
  align-items: flex-end;
  margin-bottom: 20px;
  padding: 20px;
  background-color: #f8f9fa;
  border-radius: 8px;
  border: 1px solid #e0e0e0;
}

.filter-group {
  display: flex;
  flex-direction: column;
  gap: 8px;
  flex: 1;
}

.filter-group label {
  font-weight: 600;
  color: #333;
  font-size: 14px;
}

.filter-select {
  padding: 10px;
  border: 1px solid #ccc;
  border-radius: 4px;
  font-size: 14px;
  background-color: white;
  cursor: pointer;
  transition: border-color 0.2s;
}

.filter-select:hover {
  border-color: #0087DC;
}

.filter-select:focus {
  outline: none;
  border-color: #0087DC;
  box-shadow: 0 0 0 2px rgba(0, 135, 220, 0.1);
}

.clear-filters-btn {
  padding: 10px 20px;
  background-color: #6c757d;
  color: white;
  border: none;
  border-radius: 4px;
  font-size: 14px;
  font-weight: 500;
  cursor: pointer;
  transition: background-color 0.2s;
  white-space: nowrap;
}

.clear-filters-btn:hover:not(:disabled) {
  background-color: #5a6268;
}

.clear-filters-btn:disabled {
  background-color: #ccc;
  cursor: not-allowed;
  opacity: 0.6;
}

.results-info {
  margin-bottom: 15px;
  padding: 10px;
  background-color: #e7f3ff;
  border-left: 4px solid #0087DC;
  border-radius: 4px;
}

.results-info p {
  margin: 0;
  color: #333;
  font-size: 14px;
}

.fault-history-table {
  overflow-x: auto;
}

.fault-history-table table {
  width: 100%;
  border-collapse: collapse;
  background-color: #fff;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.fault-history-table th {
  background-color: #0087DC;
  color: white;
  padding: 12px;
  text-align: left;
  font-weight: 600;
}

.fault-history-table td {
  padding: 12px;
  border-bottom: 1px solid #e0e0e0;
}

.fault-history-table tr:hover {
  background-color: #f5f5f5;
}

.fault-history-table tr:last-child td {
  border-bottom: none;
}

/* Export Section */
.export-section {
  padding: 20px;
}

.export-header {
  margin-bottom: 30px;
}

.export-header h3 {
  margin: 0 0 10px 0;
  color: #333;
  font-size: 24px;
}

.export-description {
  margin: 0;
  color: #666;
  font-size: 14px;
}

.export-info {
  margin: 20px 0;
  padding: 15px;
  background-color: #e7f3ff;
  border-left: 4px solid #0087DC;
  border-radius: 4px;
}

.export-stats {
  margin: 0;
}

.export-stats p {
  margin: 5px 0;
  color: #333;
  font-size: 14px;
}

.filter-info {
  color: #666;
  font-style: italic;
}

.export-actions {
  margin-top: 30px;
  text-align: center;
  padding: 30px;
  background-color: #f8f9fa;
  border-radius: 8px;
  border: 2px dashed #0087DC;
}

.export-btn {
  padding: 15px 40px;
  background-color: #0087DC;
  color: white;
  border: none;
  border-radius: 6px;
  font-size: 16px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s;
  display: inline-flex;
  align-items: center;
  gap: 10px;
  box-shadow: 0 2px 4px rgba(0, 135, 220, 0.2);
}

.export-btn:hover:not(:disabled) {
  background-color: #006bb3;
  transform: translateY(-2px);
  box-shadow: 0 4px 8px rgba(0, 135, 220, 0.3);
}

.export-btn:disabled {
  background-color: #ccc;
  cursor: not-allowed;
  opacity: 0.6;
}

.export-icon {
  font-size: 20px;
}

.export-hint {
  margin: 15px 0 0 0;
  color: #666;
  font-size: 13px;
}

</style>
