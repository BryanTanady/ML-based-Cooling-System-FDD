import axios from 'axios'

const sanitizeBaseUrl = (value) => {
  if (!value || typeof value !== 'string') return ''
  return value.trim().replace(/\/+$/, '')
}

const getFallbackBackendUrl = () => {
  if (typeof window === 'undefined') return 'http://127.0.0.1:8001'
  return `http://${window.location.hostname}:8001`
}

const getCandidateBaseUrls = () => {
  const candidates = []
  const envUrl = sanitizeBaseUrl(import.meta.env.VITE_BACKEND_URL)
  const fallbackUrl = sanitizeBaseUrl(getFallbackBackendUrl())

  if (envUrl) candidates.push(envUrl)
  if (fallbackUrl && !candidates.includes(fallbackUrl)) {
    candidates.push(fallbackUrl)
  }

  if (typeof window !== 'undefined') {
    const altHost = window.location.hostname === 'localhost' ? '127.0.0.1' : 'localhost'
    const altUrl = sanitizeBaseUrl(`http://${altHost}:8001`)
    if (altUrl && !candidates.includes(altUrl)) {
      candidates.push(altUrl)
    }
  }

  return candidates.length > 0 ? candidates : ['http://127.0.0.1:8001']
}

export function useApi() {
  const baseUrls = getCandidateBaseUrls()
  const api = axios.create({
    baseURL: baseUrls[0],
  })

  const requestWithFallback = async (config) => {
    let lastError = null

    for (let i = 0; i < baseUrls.length; i += 1) {
      try {
        return await api.request({ ...config, baseURL: baseUrls[i] })
      } catch (error) {
        lastError = error
        const isNetworkError = !error?.response
        const hasMoreCandidates = i < baseUrls.length - 1
        if (!(isNetworkError && hasMoreCandidates)) {
          throw error
        }
      }
    }

    throw lastError
  }
  
  const getFaultHistory = async () => {
    try {
      const response = await requestWithFallback({ method: 'get', url: '/api/db/fault_history' })
      return response.data
    } catch (error) {
      console.error('Error fetching fault history:', error)
      throw error
    }
  }

  const getRawAlerts = async () => {
    try {
      const response = await requestWithFallback({ method: 'get', url: '/api/db/raw_alerts' })
      return response.data
    } catch (error) {
      console.error('Error fetching raw alerts:', error)
      throw error
    }
  }
  const acknowledgeFaultPeriod = async (id, acknowledged_at) => {
    try {
      const response = await requestWithFallback({
        method: 'post',
        url: '/api/fault_periods/ack',
        data: { id, acknowledged_at },
      })
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
