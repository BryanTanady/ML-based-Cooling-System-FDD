<template>
  <div class="detected-faults">
    <h3>Pending Acknowledgement  <span v-if="alerts.length > 0" id="alert-count">({{ alerts.length }})</span></h3>
    <div class="alerts-list">
      <div v-if="alerts.length === 0" class="no-alerts">
        No alerts received yet
      </div>
      <div 
        v-for="(alert, index) in alerts" 
        :key="index" 
        class="alert-item"
        :class="alert.severity"
      >
        <div class="alert-header">
          <span class="alert-asset">{{ alert.asset_id }}</span>
          <span class="alert-severity">{{ alert.severity.toUpperCase() }}</span>
        </div>
        <div class="alert-message">{{ alert.message }}</div>
        <div class="alert-footer">
          <span class="alert-count-badge">Total: {{ alert.count }}</span>
          <span class="alert-time">{{ formatTime(alert.ts) }}</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
defineProps({
  alerts: {
    type: Array,
    default: () => []
  }
})

const formatTime = (ts) => {
  if (!ts) return 'Unknown time'
  const timestamp = ts < 10000000000 ? ts * 1000 : ts
  return new Date(timestamp).toLocaleString()
}
</script>

<style scoped>
.detected-faults{
  padding: 20px;
  border-radius: 8px;
  border: 1px solid black;
  height: 50%;
}

.detected-faults h3 {
  margin: 0 0 16px 0;
  color: #333;
}

.alert-count {
  color: #f44336;
  font-weight: normal;
  font-size: 0.9em;
}

.alerts-list {
  display: flex;
  flex-direction: column;

  gap: 12px;
  max-height: 600px;
  overflow-y: auto;
}

.no-alerts {
  text-align: center;
  color: #999;
  padding: 40px;
  font-style: italic;
}

.alert-item {
  background: #ffffff;
  border-left: 4px solid #0087DC;
  border-radius: 6px;
  padding: 12px 16px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
  border: 1px solid black;
  transition: transform 0.2s, box-shadow 0.2s;
}

.alert-item:hover {
  transform: translateX(2px);
  box-shadow: 0 4px 8px rgba(0,0,0,0.15);
}

.alert-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
}

.alert-asset {
  font-weight: 600;
  color: #222;
  font-size: 16px;
}

.alert-severity {
  padding: 4px 8px;
  border-radius: 4px;
  font-size: 12px;
  font-weight: 600;
  text-transform: uppercase;
}

.alert-message {
  color: #555;
  margin-bottom: 8px;
  line-height: 1.4;
}

.alert-footer {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 12px;
  color: #888;
  padding-top: 8px;
  border-top: 1px solid #eee;
}

.alert-count-badge {
  background: #f0f0f0;
  padding: 2px 8px;
  border-radius: 12px;
}

.alert-time {
  color: #999;
}

/* Severity color coding */
.alert-item.critical {
  border-left-color: #f44336;
}

.alert-item.critical .alert-severity {
  background-color: #f44336;
  color: white;
}

.alert-item.major {
  border-left-color: #ff9800;
}

.alert-item.major .alert-severity {
  background-color: #ff9800;
  color: white;
}

.alert-item.minor {
  border-left-color: #4caf50;
}

.alert-item.minor .alert-severity {
  background-color: #4caf50;
  color: white;
}

.alert-item.info {
  border-left-color: #0087DC;
}

.alert-item.info .alert-severity {
  background-color: #0087DC;
  color: white;
}
</style>

