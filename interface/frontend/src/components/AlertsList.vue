<template>
  <div class="detected-faults">
    <div class="alerts-tabs">
      <button
        class="tab-button"
        :class="{ active: activeTab === 'pending' }"
        @click="activeTab = 'pending'"
      >
        Pending Acknowledgement
      </button>
      <button
        class="tab-button"
        :class="{ active: activeTab === 'graph' }"
        @click="activeTab = 'graph'"
      >
        Graph View
      </button>
    </div>

    <template v-if="activeTab === 'pending'">
      <h3>Pending Acknowledgement <span v-if="alerts.length > 0" id="alert-count">({{ alerts.length }})</span></h3>
      <div class="alerts-list">
        <div v-if="alerts.length === 0" class="no-alerts">
          No alerts received yet
        </div>
        <div
          v-for="(alert, index) in alerts"
          :key="`pending-${alert.id}-${index}`"
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
    </template>

    <template v-else>
      <h3>Graph View</h3>
        <div class="graph-panel">
          <div class="graph-controls">
          <div class="graph-control-group">
            <label for="observation-window">Observation Length</label>
            <select id="observation-window" v-model.number="observationWindowSec">
              <option
                v-for="option in observationWindowOptions"
                :key="option.value"
                :value="option.value"
              >
                {{ option.label }}
              </option>
            </select>
          </div>

          <div class="graph-control-group">
            <label for="line-connect-gap">Line Connect Gap</label>
            <div class="graph-control-inline">
              <input
                id="line-connect-gap"
                v-model.number="lineConnectGapMs"
                type="number"
                min="0"
                step="50"
              >
              <span class="graph-control-unit">ms</span>
            </div>
          </div>
        </div>

        <div v-if="chartSeries.length === 0" class="graph-legend-empty">
          No confidence series in the selected window.
        </div>

        <div v-else class="graph-legend">
          <div v-for="series in chartSeries" :key="series.conditionName" class="legend-item">
            <span class="legend-swatch" :style="{ backgroundColor: series.color }"></span>
            <span>{{ series.label }}</span>
          </div>
        </div>

        <div class="chart-container">
          <svg
            class="confidence-chart"
            :viewBox="`0 0 ${chartWidth} ${chartHeight}`"
            preserveAspectRatio="none"
          >
            <g>
              <line
                v-for="tick in yTicks"
                :key="`grid-${tick.value}`"
                :x1="chartPadding.left"
                :x2="chartWidth - chartPadding.right"
                :y1="tick.y"
                :y2="tick.y"
                class="grid-line"
              />
            </g>

            <line
              :x1="chartPadding.left"
              :x2="chartPadding.left"
              :y1="chartPadding.top"
              :y2="chartHeight - chartPadding.bottom"
              class="axis-line"
            />
            <line
              :x1="chartPadding.left"
              :x2="chartWidth - chartPadding.right"
              :y1="chartHeight - chartPadding.bottom"
              :y2="chartHeight - chartPadding.bottom"
              class="axis-line"
            />

            <g v-for="tick in yTicks" :key="`ylabel-${tick.value}`">
              <text
                :x="chartPadding.left - 10"
                :y="tick.y + 4"
                class="tick-label y-tick"
              >
                {{ tick.label }}
              </text>
            </g>

            <g v-for="tick in xTicks" :key="`xlabel-${tick.value}`">
              <text
                :x="tick.x"
                :y="chartHeight - chartPadding.bottom + 20"
                class="tick-label x-tick"
              >
                {{ tick.label }}
              </text>
            </g>

            <line
              v-if="chartSeries.length === 0"
              :x1="chartPadding.left"
              :x2="chartWidth - chartPadding.right"
              :y1="scaleY(0.5)"
              :y2="scaleY(0.5)"
              class="empty-series-line"
            />

            <text
              v-if="chartSeries.length === 0"
              :x="chartWidth / 2"
              :y="scaleY(0.5) - 10"
              class="empty-series-label"
            >
              Waiting for confidence data
            </text>

            <g v-for="series in chartSeries" :key="series.conditionName">
              <polyline
                v-for="segment in series.lineSegments"
                :key="segment.id"
                :points="segment.polylinePoints"
                fill="none"
                :stroke="series.color"
                stroke-width="2.5"
                stroke-linecap="round"
                stroke-linejoin="round"
              />
              <circle
                v-for="(point, pointIndex) in series.points"
                :key="`${series.conditionName}-${point.tsMs}-${pointIndex}`"
                :cx="point.x"
                :cy="point.y"
                r="3"
                :fill="series.color"
              />
            </g>

            <text :x="chartWidth / 2" :y="chartHeight - 6" class="axis-label">Relative Time (s)</text>
            <text
              :x="18"
              :y="chartHeight / 2"
              class="axis-label"
              :transform="`rotate(-90 18 ${chartHeight / 2})`"
            >
              Confidence
            </text>
          </svg>
        </div>
      </div>
    </template>
  </div>
</template>

<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'

const props = defineProps({
  alerts: {
    type: Array,
    default: () => [],
  },
  rawAlerts: {
    type: Array,
    default: () => [],
  },
  acknowledgeAlert: {
    type: Function,
    required: true,
  },
})

const activeTab = ref('pending')
const observationWindowSec = ref(30)
const lineConnectGapMs = ref(600)
const observationWindowOptions = [
  { label: '10 seconds', value: 10 },
  { label: '30 seconds', value: 30 },
  { label: '1 minute', value: 60 },
]
const nowMs = ref(Date.now())
let clockTimer = null

onMounted(() => {
  clockTimer = setInterval(() => {
    nowMs.value = Date.now()
  }, 1000)
})

onBeforeUnmount(() => {
  if (clockTimer) clearInterval(clockTimer)
})

const chartWidth = 900
const chartHeight = 240
const chartPadding = {
  top: 16,
  right: 22,
  bottom: 42,
  left: 62,
}

const FAULT_COLORS = {
  BLOCKED_AIRFLOW: '#0052cc',
  INTERFERENCE: '#2e7d32',
  IMBALANCE: '#ef6c00',
  UNKNOWN: '#d32f2f',
}

function normalizeFaultCode(nameOrCode) {
  if (!nameOrCode || typeof nameOrCode !== 'string') return 'UNKNOWN'
  const key = nameOrCode.toUpperCase().replace(/\s+/g, '_').replace(/-+/g, '_')
  if (key.includes('BLOCKED') || key === 'BLOCKED_AIRFLOW' || key === 'FAN_BLOCKED') return 'BLOCKED_AIRFLOW'
  if (key.includes('INTERFERE') || key.includes('BLADE') || key === 'INTERFERENCE' || key === 'BLADE_ISSUE') return 'INTERFERENCE'
  if (key.includes('IMBALANCE') || key.includes('POWER') || key.includes('ELECTR') || key === 'IMBALANCE' || key === 'POWER_ISSUE') return 'IMBALANCE'
  return key || 'UNKNOWN'
}

const getFaultClass = (alert) => {
  const code = alert.fault_type || alert.condition_name
  return normalizeFaultCode(code)
}

const getFaultColor = (conditionName) => {
  const normalized = normalizeFaultCode(conditionName)
  return FAULT_COLORS[normalized] || FAULT_COLORS.UNKNOWN
}

const formatConditionLabel = (conditionName) => {
  const normalized = normalizeFaultCode(conditionName)
  return normalized
    .replace(/_/g, ' ')
    .toLowerCase()
    .replace(/\b\w/g, (c) => c.toUpperCase())
}

const formatTime = (ts) => {
  if (!ts) return 'Unknown time'
  const timestamp = typeof ts === 'string' ? new Date(ts).getTime() : (ts < 10000000000 ? ts * 1000 : ts)
  return new Date(timestamp).toLocaleString()
}

const normalizeTimestampMs = (ts) => {
  if (!Number.isFinite(ts)) return null
  return ts < 10000000000 ? ts * 1000 : ts
}

const parseConfidence = (value) => {
  if (!Number.isFinite(value)) return null
  return Math.max(0, Math.min(1, value))
}

const chartPoints = computed(() => {
  return props.rawAlerts
    .map((alert) => {
      const tsMs = normalizeTimestampMs(Number(alert.ts))
      const confidence = parseConfidence(Number(alert.confidence))
      if (tsMs === null || confidence === null) return null
      return {
        tsMs,
        confidence,
        conditionName: normalizeFaultCode(alert.condition_name || alert.message),
      }
    })
    .filter((point) => point !== null)
    .sort((a, b) => a.tsMs - b.tsMs)
})

const timeAnchorMs = computed(() => {
  if (chartPoints.value.length === 0) {
    return nowMs.value
  }
  const latestPointTsMs = chartPoints.value[chartPoints.value.length - 1].tsMs
  return Math.max(latestPointTsMs, nowMs.value)
})

const visibleChartPoints = computed(() => {
  const windowStartTsMs = timeAnchorMs.value - observationWindowSec.value * 1000
  return chartPoints.value.filter((point) => point.tsMs >= windowStartTsMs)
})

const normalizedLineConnectGapMs = computed(() => {
  if (!Number.isFinite(lineConnectGapMs.value)) return 0
  return Math.max(0, lineConnectGapMs.value)
})

const timeDomain = computed(() => {
  const max = timeAnchorMs.value
  const min = max - observationWindowSec.value * 1000
  const span = Math.max(1, max - min)
  return { min, max, span }
})

const scaleX = (tsMs) => {
  const plotWidth = chartWidth - chartPadding.left - chartPadding.right
  return chartPadding.left + ((tsMs - timeDomain.value.min) / timeDomain.value.span) * plotWidth
}

const scaleY = (confidence) => {
  const plotHeight = chartHeight - chartPadding.top - chartPadding.bottom
  return chartPadding.top + (1 - confidence) * plotHeight
}

const chartSeries = computed(() => {
  const grouped = new Map()
  let previousConditionName = null
  let previousPointTsMs = null
  let runId = -1

  visibleChartPoints.value.forEach((point) => {
    const gapFromPreviousMs = previousPointTsMs === null ? 0 : point.tsMs - previousPointTsMs
    if (
      point.conditionName !== previousConditionName
      || gapFromPreviousMs > normalizedLineConnectGapMs.value
    ) {
      runId += 1
    }
    previousConditionName = point.conditionName
    previousPointTsMs = point.tsMs

    if (!grouped.has(point.conditionName)) {
      grouped.set(point.conditionName, {
        conditionName: point.conditionName,
        label: formatConditionLabel(point.conditionName),
        color: getFaultColor(point.conditionName),
        segmentsByRun: new Map(),
      })
    }

    const renderedPoint = {
      ...point,
      x: scaleX(point.tsMs),
      y: scaleY(point.confidence),
    }

    const series = grouped.get(point.conditionName)
    if (!series.segmentsByRun.has(runId)) {
      series.segmentsByRun.set(runId, [])
    }
    series.segmentsByRun.get(runId).push(renderedPoint)
  })

  return Array.from(grouped.values())
    .map((series) => {
      const segments = Array.from(series.segmentsByRun.entries()).map(([segmentRunId, points]) => ({
        id: `${series.conditionName}-${segmentRunId}`,
        points,
        polylinePoints: points.map((point) => `${point.x},${point.y}`).join(' '),
      }))

      return {
        conditionName: series.conditionName,
        label: series.label,
        color: series.color,
        points: segments.flatMap((segment) => segment.points),
        segments,
        lineSegments: segments.filter((segment) => segment.points.length > 1),
      }
    })
    .sort((a, b) => a.label.localeCompare(b.label))
})

const yTicks = computed(() => {
  return [0, 0.25, 0.5, 0.75, 1].map((value) => ({
    value,
    y: scaleY(value),
    label: `${(value * 100).toFixed(0)}%`,
  }))
})

const formatAxisTime = (tsMs) => {
  const elapsedSec = Math.max(0, (tsMs - timeDomain.value.min) / 1000)
  return `${Math.round(elapsedSec)}s`
}

const xTicks = computed(() => {
  const min = timeDomain.value.min
  const max = timeDomain.value.max
  const mid = min + timeDomain.value.span / 2
  const values = [min, mid, max]

  return values.map((value) => ({
    value,
    x: scaleX(value),
    label: formatAxisTime(value),
  }))
})
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

.alerts-tabs {
  display: flex;
  gap: 10px;
  margin-bottom: 14px;
}

.tab-button {
  padding: 8px 14px;
  border: 1px solid #c9d7e6;
  border-radius: 6px;
  background: #edf3f9;
  color: #2b3a4a;
  font-size: 15px;
  font-weight: 600;
  cursor: pointer;
}

.tab-button.active {
  border-color: #0087dc;
  background: #0087dc;
  color: #fff;
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

.alert-item.BLOCKED_AIRFLOW {
  border-left-color: #0052cc;
}

.alert-item.INTERFERENCE {
  border-left-color: #2e7d32;
}

.alert-item.IMBALANCE {
  border-left-color: #ef6c00;
}

.alert-item.UNKNOWN {
  border-left-color: #d32f2f;
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

.graph-panel {
  display: flex;
  flex-direction: column;
  gap: 10px;
  flex: 1 1 0;
  min-height: 0;
}

.graph-controls {
  display: flex;
  align-items: flex-end;
  gap: 14px;
  flex-wrap: wrap;
}

.graph-control-group {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.graph-controls label {
  font-size: 14px;
  font-weight: 600;
  color: #334155;
}

.graph-control-inline {
  display: flex;
  align-items: center;
  gap: 6px;
}

.graph-controls select {
  border: 1px solid #cbd5e1;
  border-radius: 6px;
  background: #fff;
  color: #1e293b;
  font-size: 15px;
  padding: 6px 10px;
}

.graph-controls input {
  border: 1px solid #cbd5e1;
  border-radius: 6px;
  background: #fff;
  color: #1e293b;
  font-size: 15px;
  padding: 6px 10px;
  width: 96px;
}

.graph-control-unit {
  color: #475569;
  font-size: 14px;
  font-weight: 600;
}

.graph-legend {
  display: flex;
  gap: 14px;
  flex-wrap: wrap;
}

.graph-legend-empty {
  color: #667085;
  font-size: 13px;
}

.legend-item {
  display: flex;
  align-items: center;
  gap: 7px;
  font-size: 16px;
  color: #334155;
}

.legend-swatch {
  width: 14px;
  height: 14px;
  border-radius: 3px;
  display: inline-block;
}

.chart-container {
  background: #fff;
  border-radius: 8px;
  border: 1px solid #dbe7f3;
  padding: 10px 12px 6px;
  flex: 1 1 0;
  min-height: 260px;
}

.confidence-chart {
  width: 100%;
  height: 100%;
  min-height: 260px;
}

.grid-line {
  stroke: #edf2f7;
  stroke-width: 1;
}

.axis-line {
  stroke: #64748b;
  stroke-width: 1.2;
}

.tick-label {
  fill: #64748b;
  font-size: 14px;
}

.y-tick {
  text-anchor: end;
}

.x-tick {
  text-anchor: middle;
}

.axis-label {
  fill: #334155;
  font-size: 16px;
  font-weight: 700;
}

.empty-series-line {
  stroke: #94a3b8;
  stroke-width: 2;
  stroke-dasharray: 7 6;
}

.empty-series-label {
  fill: #64748b;
  font-size: 12px;
  text-anchor: middle;
}

@media (max-width: 900px) {
  .alert-footer {
    flex-wrap: wrap;
    gap: 6px;
  }
}
</style>
