<template>
  <div class="system-status">
    <h3>System Status / Active Faults</h3>
    <div class="fault-breakdown">
      <div class="fault-item BLOCKED_AIRFLOW" :class="{ highlight: highlightedFaultCode === 'BLOCKED_AIRFLOW' }">
        <span class="fault-label">BLOCKED_AIRFLOW</span>
        <span class="fault-count">{{ faultCounts.Fan_Blocked }}</span>
      </div>  
      <div class="fault-item INTERFERENCE" :class="{ highlight: highlightedFaultCode === 'INTERFERENCE' }">
        <span class="fault-label">INTERFERENCE</span>
        <span class="fault-count">{{ faultCounts.Fan_Interference }}</span>
      </div>  
      <div class="fault-item IMBALANCE" :class="{ highlight: highlightedFaultCode === 'IMBALANCE' }">
        <span class="fault-label">IMBALANCE</span>
        <span class="fault-count">{{ faultCounts.Fan_Imbalance }}</span>
      </div>  
      <div class="fault-item UNKNOWN" :class="{ highlight: highlightedFaultCode === 'UNKNOWN' }">
        <span class="fault-label">UNKNOWN</span>
        <span class="fault-count">{{ faultCounts.Unknown }}</span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, ref, onMounted, onBeforeUnmount } from 'vue'

// Count alerts by fault type
const props = defineProps({
  alerts: {
    type: Array,
    default: () => []
  },
  rawAlerts: {
    type: Array,
    default: () => []
  }
})

const faultCounts = computed(() => {
  const counts = { Fan_Blocked: 0, Fan_Interference: 0, Fan_Imbalance: 0, Unknown: 0 }
  props.alerts.forEach(alert => {
    const code = alert?.fault_type_code || alert?.fault_type
    if (code === 'BLOCKED_AIRFLOW') {
      counts.Fan_Blocked++
    } else if (code === 'INTERFERENCE' || code === 'BLADE_ISSUE') {
      counts.Fan_Interference++
    } else if (code === 'IMBALANCE' || code === 'POWER_ISSUE') {
      counts.Fan_Imbalance++
    } else {
      counts.Unknown++
    }
  })
  return counts
})

const toFaultCode = (alert) => {
  const rawCode = (
    alert?.fault_type_code
    || alert?.fault_type
    || alert?.condition_name
    || alert?.message
    || ''
  ).toString()

  const key = rawCode
    .toUpperCase()
    .replace(/\s+/g, '_')
    .replace(/-+/g, '_')

  if (key.includes('BLOCKED') || key === 'BLOCKED_AIRFLOW' || key === 'FAN_BLOCKED') {
    return 'BLOCKED_AIRFLOW'
  }
  if (key.includes('INTERFERE') || key.includes('BLADE') || key === 'INTERFERENCE' || key === 'BLADE_ISSUE') {
    return 'INTERFERENCE'
  }
  if (
    key.includes('IMBALANCE')
    || key.includes('POWER')
    || key.includes('ELECTR')
    || key === 'IMBALANCE'
    || key === 'POWER_ISSUE'
  ) {
    return 'IMBALANCE'
  }
  return 'UNKNOWN'
}

const ONE_SECOND_MS = 1000
const nowMs = ref(Date.now())
let nowTicker = null

const toMillis = (ts) => {
  if (typeof ts === 'number') {
    return ts > 10_000_000_000 ? ts : ts * 1000
  }
  if (typeof ts !== 'string') return null
  const parsed = Date.parse(ts)
  return Number.isNaN(parsed) ? null : parsed
}

const highlightedFaultCode = computed(() => {
  const latestRaw = props.rawAlerts[0]
  if (!latestRaw) return ''

  const eventMs = toMillis(latestRaw.received_at_ms) ?? toMillis(latestRaw.ts)
  if (eventMs === null) return ''

  const ageMs = nowMs.value - eventMs
  if (ageMs < 0 || ageMs > ONE_SECOND_MS) return ''

  return toFaultCode(latestRaw)
})

onMounted(() => {
  nowTicker = setInterval(() => {
    nowMs.value = Date.now()
  }, 100)
})

onBeforeUnmount(() => {
  if (nowTicker) {
    clearInterval(nowTicker)
    nowTicker = null
  }
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
  transition: box-shadow 0.2s ease, background-color 0.2s ease;
}

.fault-item.highlight {
  background: #fff8e1;
  box-shadow: 0 0 0 4px rgba(255, 193, 7, 0.75), 0 0 16px rgba(255, 193, 7, 0.8);
}

.fault-item.BLOCKED_AIRFLOW {
  border-left-color: #0000FF;
}

.fault-item.INTERFERENCE {
  border-left-color: #4caf50;
}

.fault-item.IMBALANCE {
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

.fault-item.INTERFERENCE .fault-count {
  background: #4caf50;
  color: white;
}

.fault-item.IMBALANCE .fault-count {
  background: #ff9800;
  color: white;
}

.fault-item.UNKNOWN .fault-count {
  background: #f44336;
  color: white;
}
</style>
