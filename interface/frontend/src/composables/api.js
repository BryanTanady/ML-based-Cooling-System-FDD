import axios from 'axios'

export function useApi() {
  const api = axios.create({
    baseURL: import.meta.env.VITE_BACKEND_URL,
  })
  
  const getFaultHistory = async () => {
    try {
      const response = await api.get('/api/db/fault_history')
      return response.data
    } catch (error) {
      console.error('Error fetching fault history:', error)
      throw error
    }
  }

  const getRawAlerts = async () => {
    try {
      const response = await api.get('/api/db/raw_alerts')
      return response.data
    } catch (error) {
      console.error('Error fetching raw alerts:', error)
      throw error
    }
  }
  const acknowledgeFaultPeriod = async (id, acknowledged_at) => {
    try {
      const response = await api.post('/api/fault_periods/ack', { id, acknowledged_at })
      return response.data
    } catch (error) {
      console.error('Error acknowledging fault period:', error)
      throw error
    }
  }
  
  return {
    api,
    getFaultHistory,
    getRawAlerts,
    acknowledgeFaultPeriod,
  }
}
