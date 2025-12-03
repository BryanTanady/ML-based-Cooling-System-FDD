import axios from 'axios'

export function useApi() {
  const api = axios.create({
    baseURL: 'http://localhost:8000/api',
  })
  
  const getFaultHistory = async () => {
    try {
      const response = await api.get('/db/fault_history')
      return response.data
    } catch (error) {
      console.error('Error fetching fault history:', error)
      throw error
    }
  }
  
  return {
    api,
    getFaultHistory,
  }
}
