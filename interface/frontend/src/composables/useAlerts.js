import { ref, onMounted, onBeforeUnmount } from 'vue'

export function useAlerts() {
  const alerts = ref([])
  const status = ref('disconnected') // disconnected | connecting | connected | error
  const current = ref(null)
  const url = 'ws://localhost:8000/ws/alerts' //change as needed

  // websocket connection
  let ws = null
  let reconnectTimer = null //auto reconnect timer

  // close current alert
  const closeCurrent = () => { current.value = null }

  // connect to websocket
  const connect = () => {
    try {
      status.value = 'connecting'
      ws = new WebSocket(url)

      ws.onopen = () => { status.value = 'connected' }
      ws.onclose = () => {
        status.value = 'disconnected'
        reconnectTimer = setTimeout(connect, 2000)//auto reconnect every 2 seconds
      }
      ws.onerror = () => { status.value = 'error' } //error handling

      // handle incoming messages
      ws.onmessage = (evt) => {
        try {
          const raw = JSON.parse(evt.data)
          const payload = {
            asset_id: raw.asset_id || 'Unknown asset',
            message: raw.message || 'Unknown alert',
            ts: raw.ts || Date.now(),
            count: raw.count || 0,
          }
          alerts.value.unshift(payload)
          current.value = payload
        } catch (e) {
          const fallback = { asset_id: 'Unknown asset', message: String(evt.data), ts: Date.now() }
          alerts.value.unshift(fallback)
          current.value = fallback
        }
      }
    } catch (e) {
      status.value = 'error'
      console.error('WebSocket error:', e)
    }
  }
  onMounted(() => connect())
  onBeforeUnmount(() => {
    if (reconnectTimer) clearTimeout(reconnectTimer)
    if (ws) ws.close()
  })
  return {
    alerts,
    status,
    currentAlert: current,
    closeCurrent,
  }
}
