import { ref, onMounted, onBeforeUnmount } from 'vue'
import { useApi } from '@/composables/api'

export function useAlerts() {
  const { acknowledgeFaultPeriod } = useApi()

  const alerts = ref([])
  const rawAlerts = ref([])
  const MAX_RAW_ALERTS = 2000
  const status = ref('disconnected') // disconnected | connecting | connected | error
  const current = ref(null)

  const url = 'ws://localhost:8001/ws/alerts' // change as needed

  const normalizeFaultType = (rawFaultType) => {
    const s = (rawFaultType || '').toString().trim()
    const key = s
      .toUpperCase()
      .replace(/\s+/g, '_')
      .replace(/-+/g, '_')

    if (key.includes('BLOCKED') || key === 'BLOCKED_AIRFLOW' || key === 'FAN_BLOCKED') {
      return { code: 'BLOCKED_AIRFLOW', label: 'Blocked Airflow' }
    }
    if (key.includes('INTERFERE') || key.includes('BLADE') || key === 'INTERFERENCE' || key === 'BLADE_ISSUE') {
      return { code: 'INTERFERENCE', label: 'Interference' }
    }
    if (
      key.includes('IMBALANCE')
      || key.includes('POWER')
      || key.includes('ELECTR')
      || key === 'IMBALANCE'
      || key === 'POWER_ISSUE'
    ) {
      return { code: 'IMBALANCE', label: 'Imbalance' }
    }

    return { code: 'UNKNOWN', label: s || 'Unknown' }
  }

  let ws = null
  let reconnectTimer = null

  const closeCurrent = () => {
    current.value = null
  }

  const acknowledgeAlert = async (index) => {
    if (index < 0 || index >= alerts.value.length) return

    const removed = alerts.value[index]
    const currentId = current.value && current.value.id
    const removedId = removed && removed.id
    const removedWasCurrent = Boolean(currentId && removedId && currentId === removedId)

    // optimistic UI update
    alerts.value.splice(index, 1)

    if (removedWasCurrent) {
      current.value = alerts.value[0] || null
    }

    if (alerts.value.length === 0) {
      current.value = null
    }

    const faultId = removed && removed.id
    if (!faultId || faultId === 'Unknown id') return

    try {
      await acknowledgeFaultPeriod(faultId, new Date().toISOString())
    } catch (error) {
      console.error('Error acknowledging fault period:', error)

      // rollback
      alerts.value.splice(index, 0, removed)
      if (removedWasCurrent) {
        current.value = removed
      }
    }
  }

  const cleanupSocket = () => {
    if (!ws) return

    ws.onopen = null
    ws.onclose = null
    ws.onerror = null
    ws.onmessage = null
    ws = null
  }

  const scheduleReconnect = () => {
    if (reconnectTimer) return

    reconnectTimer = setTimeout(() => {
      reconnectTimer = null
      connect()
    }, 2000)
  }

  const connect = () => {
    // do not create a new socket if one already exists and is active
    if (
      ws &&
      (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)
    ) {
      return
    }

    try {
      status.value = 'connecting'
      ws = new WebSocket(url)

      ws.onopen = () => {
        status.value = 'connected'

        if (reconnectTimer) {
          clearTimeout(reconnectTimer)
          reconnectTimer = null
        }
      }

      ws.onclose = () => {
        status.value = 'disconnected'
        cleanupSocket()
        scheduleReconnect()
      }

      ws.onerror = (error) => {
        status.value = 'error'
        console.error('WebSocket error:', error)
      }

      ws.onmessage = (evt) => {
        try {
          const raw = JSON.parse(evt.data)

          if (raw.type === 'raw_alert') {
            const entry = {
              asset_id: raw.asset_id ?? '',
              condition_id: raw.condition_id,
              condition_name: raw.condition_name ?? '',
              message: raw.message ?? '',
              confidence: raw.confidence,
              ts: raw.ts,
              received_at_ms: Date.now(),
            }
            rawAlerts.value.unshift(entry)
            if (rawAlerts.value.length > MAX_RAW_ALERTS) {
              rawAlerts.value = rawAlerts.value.slice(0, MAX_RAW_ALERTS)
            }
            return
          }

          if (raw.type === 'fault_period' || !raw.type) {
            const payload = {
              type: raw.type || 'fault_period',
              id: raw.id || 'Unknown id',
              asset_id: raw.asset_id || 'Unknown asset',
              fault_type: raw.fault_type || 'UNKNOWN',
              start_ts: raw.start_ts || 'Unknown start time',
            }
            alerts.value.unshift(payload)
            if (!current.value) {
              current.value = payload
            }
          }
        } catch (e) {
          console.error('Error parsing WebSocket message:', e)
        }
      }
    } catch (e) {
      status.value = 'error'
      console.error('WebSocket creation error:', e)
      scheduleReconnect()
    }
  }

  onMounted(() => {
    connect()
  })

  onBeforeUnmount(() => {
    if (reconnectTimer) {
      clearTimeout(reconnectTimer)
      reconnectTimer = null
    }

    if (ws) {
      ws.close()
      cleanupSocket()
    }
  })

  return {
    alerts,
    rawAlerts,
    status,
    currentAlert: current,
    closeCurrent,
    acknowledgeAlert,
    connect,
  }
}
