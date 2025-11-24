<script setup>
import { ref } from 'vue'
import Button from './components/Button.vue'
import AlertsList from './components/AlertsList.vue'
import SystemStatus from './components/SystemStatus.vue'
import { useAlerts } from '@/composables/useAlerts'
const { alerts, status } = useAlerts()

const currentSection = ref('Overview')

const sections = [
  'Overview',
  'Fault History',
  'Maintenance History',
  'Model Health',
  'Export',
  'Help'
]

const setSection = (section) => {
  currentSection.value = section
}
</script>

<template>
  <div class="container">
    <div class="nav-bar">
      <h2 id="company-logo">Company Logo</h2>
      <div class="nav-bar-content">
        <Button 
          v-for="section in sections" 
          :key="section"
          :label="section"
          :is-active="currentSection === section"
          @click="setSection(section)"
        />
      </div>
    </div>
    <div class="main-content" >
      <div class="main-content-header">
        <h2>{{ currentSection }}</h2>
      </div>
      <div class="section-context">
        <!-- Overview Section -->
        <template v-if="currentSection === 'Overview'">
          <AlertsList :alerts="alerts" class="alerts-list"/>
          <SystemStatus :alerts="alerts" class="system-status"/>
        </template>
        
        <!-- Asset Management Section -->
        <template v-else-if="currentSection === 'Asset Management'">
          <div class="section-placeholder">
            <h3>Asset Management</h3>
            <p>Asset management content will be displayed here.</p>
          </div>
        </template>
        
        <!-- Fault History Section -->
        <template v-else-if="currentSection === 'Fault History'">
          <div class="section-placeholder">
            <h3>Fault History</h3>
            <p>Fault history content will be displayed here.</p>
          </div>
        </template>
        
        <!-- Maintenance History Section -->
        <template v-else-if="currentSection === 'Maintenance History'">
          <div class="section-placeholder">
            <h3>Maintenance History</h3>
            <p>Maintenance history content will be displayed here.</p>
          </div>
        </template>
        
        <!-- Model Health Section -->
        <template v-else-if="currentSection === 'Model Health'">
          <div class="section-placeholder">
            <h3>Model Health</h3>
            <p>Model health content will be displayed here.</p>
          </div>
        </template>
        
        <!-- Export Section -->
        <template v-else-if="currentSection === 'Export'">
          <div class="section-placeholder">
            <h3>Export</h3>
            <p>Export functionality will be displayed here.</p>
          </div>
        </template>
        
        <!-- Help Section -->
        <template v-else-if="currentSection === 'Help'">
          <div class="section-placeholder">
            <h3>Help</h3>
            <p>Help and documentation will be displayed here.</p>
          </div>
        </template>
      </div>
    </div>
  </div>
</template>

<style scoped>
.container {
  display: flex;
  flex-direction: row;
  width: 100%;
  height: 100vh;
  background-color:#F2F2F2  ;
  margin: 0;
  padding: 0;
}
 /* Nav Bar  styles */
.nav-bar {
  width: 200px;
  background-color: #FFFFFF;
  padding: 20px;
}
#company-logo {
  font-size: 24px;
  font-weight: bold;
  text-align: center;
  border: 2px solid #FFFFFF;
}

.nav-bar-content {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

 /* Main Content styles */
.main-content {
  flex: 1;
  display: flex;
  flex-direction: column;
  background-color: #FFFFFF;
  margin: 20px;
  padding: 20px;
  border: 2px solid #0087DC;
  border-radius: 10px;
}

.main-content-header {
  padding: 16px 0;
  height: 5%;
  border-bottom: 2px solid #0087DC;
  margin-bottom: 20px;
}

.main-content-header h2 {
  margin: 0;
  color: #333;
  font-size: 28px;
}

.section-context {
  flex: 1;
  height: 95%;
}

.section-placeholder {
  padding: 40px;
  text-align: center;
  color: #666;
}

.section-placeholder h3 {
  margin: 0 0 16px 0;
  color: #333;
  font-size: 24px;
}

.section-placeholder p {
  margin: 0;
  font-size: 16px;
}

</style>
