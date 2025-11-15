<template>
  <div class="system-status">
    <h3>System Status / Active Faults</h3>
    <div class="status-content">
      <div class="severity-breakdown">
        <div class="severity-item critical">
          <span class="severity-label">Critical</span>
          <span class="severity-count">{{ severityCounts.critical }}</span>
        </div>
        <div class="severity-item major">
          <span class="severity-label">Major</span>
          <span class="severity-count">{{ severityCounts.major }}</span>
        </div>
        <div class="severity-item minor">
          <span class="severity-label">Minor</span>
          <span class="severity-count">{{ severityCounts.minor }}</span>
        </div>
        <div class="severity-item info">
          <span class="severity-label">Info</span>
          <span class="severity-count">{{ severityCounts.info }}</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  alerts: {
    type: Array,
    default: () => []
  }
})

// Count alerts by severity
const severityCounts = computed(() => {
  const counts = { critical: 0, major: 0, minor: 0, info: 0 }
  props.alerts.forEach(alert => {
    const severity = (alert.severity || 'info').toLowerCase()
    if (counts.hasOwnProperty(severity)) {
      counts[severity]++
    } else {
      counts.info++
    }
  })
  return counts
})

// Calculate criticality percentage (0-100)
// Critical = 100 points, Major = 50, Minor = 20, Info = 5
const criticalityPercentage = computed(() => {
  if (props.alerts.length === 0) return 0
  
  const weights = {
    critical: 100,
    major: 50,
    minor: 20,
    info: 5
  }
  
  let totalScore = 0
  let maxPossibleScore = 0
  
  Object.keys(severityCounts.value).forEach(severity => {
    const count = severityCounts.value[severity]
    const weight = weights[severity] || 5
    totalScore += count * weight
    maxPossibleScore += count * 100 // Assume all could be critical
  })
  
  // If no alerts, return 0, otherwise calculate percentage
  if (maxPossibleScore === 0) return 0
  
  // Normalize to 0-100 scale
  const percentage = Math.min(100, (totalScore / maxPossibleScore) * 100)
  return Math.round(percentage)
})

// Determine status level and label
const statusLevel = computed(() => {
  const percentage = criticalityPercentage.value
  if (percentage >= 80) return 'critical'
  if (percentage >= 50) return 'major'
  if (percentage >= 20) return 'minor'
  return 'healthy'
})

const statusLabel = computed(() => {
  const labels = {
    critical: 'Critical Risk',
    major: 'High Risk',
    minor: 'Moderate Risk',
    healthy: 'Healthy'
  }
  return labels[statusLevel.value] || 'Unknown'
})
</script>

<style scoped>
.system-status {
  padding: 20px;
  background: #f9f9f9;
  border-radius: 8px;
  margin-top: 20px;
  border: 1px solid black;
}

.system-status h3 {
  margin: 0 0 20px 0;
  color: #333;
}

.status-content {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.status-indicator {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 12px;
}

.status-circle {
  width: 120px;
  height: 120px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  border: 8px solid;
  transition: all 0.3s ease;
}

.status-circle.healthy {
  border-color: #4caf50;
  background: rgba(76, 175, 80, 0.1);
}

.status-circle.minor {
  border-color: #4caf50;
  background: rgba(76, 175, 80, 0.1);
}

.status-circle.major {
  border-color: #ff9800;
  background: rgba(255, 152, 0, 0.1);
}

.status-circle.critical {
  border-color: #f44336;
  background: rgba(244, 67, 54, 0.1);
  animation: pulse 2s infinite;
}

@keyframes pulse {
  0%, 100% {
    box-shadow: 0 0 0 0 rgba(244, 67, 54, 0.4);
  }
  50% {
    box-shadow: 0 0 0 10px rgba(244, 67, 54, 0);
  }
}

.status-percentage {
  font-size: 32px;
  font-weight: bold;
}

.status-circle.healthy .status-percentage,
.status-circle.minor .status-percentage {
  color: #4caf50;
}

.status-circle.major .status-percentage {
  color: #ff9800;
}

.status-circle.critical .status-percentage {
  color: #f44336;
}

.status-label {
  font-size: 18px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 1px;
}

.status-indicator.healthy .status-label {
  color: #4caf50;
}

.status-indicator.minor .status-label {
  color: #4caf50;
}

.status-indicator.major .status-label {
  color: #ff9800;
}

.status-indicator.critical .status-label {
  color: #f44336;
}

.severity-breakdown {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 12px;
}

.severity-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 16px;
  background: white;
  border-radius: 6px;
  border-left: 4px solid;
  box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}

.severity-item.critical {
  border-left-color: #f44336;
}

.severity-item.major {
  border-left-color: #ff9800;
}

.severity-item.minor {
  border-left-color: #4caf50;
}

.severity-item.info {
  border-left-color: #0087DC;
}

.severity-label {
  font-weight: 600;
  color: #333;
}

.severity-count {
  font-size: 20px;
  font-weight: bold;
  padding: 4px 12px;
  border-radius: 12px;
  min-width: 40px;
  text-align: center;
}

.severity-item.critical .severity-count {
  background: #f44336;
  color: white;
}

.severity-item.major .severity-count {
  background: #ff9800;
  color: white;
}

.severity-item.minor .severity-count {
  background: #4caf50;
  color: white;
}

.severity-item.info .severity-count {
  background: #0087DC;
  color: white;
}
</style>

