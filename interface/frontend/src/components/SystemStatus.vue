<template>
  <div class="system-status">
    <h3>System Status / Active Faults</h3>
    <div class="fault-breakdown">
      <div
        v-for="fault in faultDefs"
        :key="fault.key"
        class="fault-item"
        :class="fault.key"
      >
        <span class="fault-label">{{ fault.label }}</span>
        <span class="fault-count">{{ faultCounts[fault.key] }}</span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  alerts: {
    type: Array,
    default: () => [],
  },
})

const faultDefs = [
  { key: 'BLOCKED_AIRFLOW', label: 'Blocked Airflow' },
  { key: 'BLADE_ISSUE', label: 'Blade Issue' },
  { key: 'POWER_ISSUE', label: 'Power Issue' },
  { key: 'UNKNOWN', label: 'Unknown' },
]

const faultCounts = computed(() => {
  const counts = {
    BLOCKED_AIRFLOW: 0,
    BLADE_ISSUE: 0,
    POWER_ISSUE: 0,
    UNKNOWN: 0,
  }
  props.alerts.forEach((alert) => {
    const key = typeof alert.condition_name === 'string' ? alert.condition_name : 'UNKNOWN'
    if (Object.prototype.hasOwnProperty.call(counts, key)) {
      counts[key] += 1
    } else {
      counts.UNKNOWN += 1
    }
  })
  return counts
})

</script>

<style scoped>
.system-status {
  padding: 20px;
  margin-top: 20px;
  background: #f9f9f9;
  border-radius: 8px;
  border: 1px solid black;
}

.system-status h3 {
  margin: 0 0 0px 0;
  color: #333;
}

.fault-breakdown {
  margin-top: 20px;
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 12px;
}

.fault-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 16px;
  background: white;
  border-radius: 6px;
  border-left: 4px solid;
  box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}

.fault-item.BLOCKED_AIRFLOW {
  border-left-color: #0000FF;
}

.fault-item.BLADE_ISSUE {
  border-left-color: #4caf50;
}

.fault-item.POWER_ISSUE {
  border-left-color: #ff9800;
}

.fault-item.UNKNOWN {
  border-left-color: #f44336;
}

.fault-label {
  font-weight: 600;
  color: #333;
}

.fault-count {
  font-size: 20px;
  font-weight: bold;
  padding: 4px 12px;
  border-radius: 12px;
  min-width: 40px;
  text-align: center;
}

.fault-item.BLOCKED_AIRFLOW .fault-count {
  background: #0000FF;
  color: white;
}

.fault-item.BLADE_ISSUE .fault-count {
  background: #4caf50;
  color: white;
}

.fault-item.POWER_ISSUE .fault-count {
  background: #ff9800;
  color: white;
}

.fault-item.UNKNOWN .fault-count {
  background: #f44336;
  color: white;
}
</style>
