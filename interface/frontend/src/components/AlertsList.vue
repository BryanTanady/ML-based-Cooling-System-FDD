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
        :class="getFaultClass(alert)"
      >
        <div class="alert-header">
          <span class="alert-asset">{{ alert.asset_id }}</span>
        </div>
        <div class="alert-message">{{ alert.fault_type }}</div>
        <div class="alert-footer">
          <span class="alert-time">{{ formatTime(alert.start_ts) }}</span>
        </div>
        <div class="alert-actions">
          <button @click="acknowledgeAlert(index)" class="acknowledge-btn">
            Acknowledge
          </button>
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
  },
  acknowledgeAlert: {
    type: Function,
    required: true
  }
})

const formatTime = (ts) => {
  if (!ts) return 'Unknown time'
  const timestamp = ts < 10000000000 ? ts * 1000 : ts
  return new Date(timestamp).toLocaleString()
}

const getFaultClass = (alert) => {
  const code = alert.fault_type || 'Unknown'
  if (code === 'BLOCKED_AIRFLOW') return 'Fan_Blocked'
  if (code === 'BLADE_ISSUE') return 'Fan_Blade_Issue'
  if (code === 'POWER_ISSUE') return 'Electrical_Fault'
  return 'Unknown'
}

</script>

<style scoped>
.detected-faults {
  padding: 20px;
  background: #f9f9f9;
  border-radius: 8px;
  border: 1px solid black;
  height: 60%;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.detected-faults h3 {
  margin: 0 0 20px 0;
  color: #333;
  flex-shrink: 0;
}

.alerts-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
  flex: 1 1 0;
  overflow-y: auto;
  overflow-x: hidden;
  min-height: 0;
  max-height: 100%;
}

.alerts-list::-webkit-scrollbar {
  width: 6px;
}

.alerts-list::-webkit-scrollbar-track {
  background: #f5f5f5;
  border-radius: 3px;
}

.alerts-list::-webkit-scrollbar-thumb {
  background: #ccc;
  border-radius: 3px;
}

.alerts-list::-webkit-scrollbar-thumb:hover {
  background: #999;
}

.alerts-list {
  scrollbar-width: thin;
  scrollbar-color: #ccc #f5f5f5;
}

.no-alerts {
  text-align: center;
  color: #999;
  padding: 40px;
  font-style: italic;
}

.alert-item {
  background: white;
  border-radius: 6px;
  padding: 12px 16px;
  border-left: 4px solid;
  box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}

.alert-item.Fan_Blocked {
  border-left-color: #0000FF;
}

.alert-item.Fan_Blade_Issue {
  border-left-color: #4caf50;
}

.alert-item.Electrical_Fault {
  border-left-color: #ff9800;
}

.alert-item.Unknown {
  border-left-color: #f44336;
}


.alert-asset {
  font-weight: 600;
  color: #333;
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

.alert-actions {
  margin-top: 12px;
  display: flex;
  justify-content: flex-end;
}

.acknowledge-btn {
  background-color: #0087DC;
  color: white;
  border: none;
  padding: 8px 16px;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
  font-weight: 500;
  transition: background-color 0.2s;
}

.acknowledge-btn:hover {
  background-color: #006ba8;
}

.acknowledge-btn:active {
  background-color: #005a8f;
}
</style>
