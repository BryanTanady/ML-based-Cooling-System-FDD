<template>
  <button 
    class="button" 
    :class="{ active: isActive }"
    @click="handleClick"
  >
    {{ label }}
  </button>
</template>

<script setup>
defineProps({
  label: String,
  isActive: {
    type: Boolean,
    default: false
  }
})

const emit = defineEmits(['click'])

const handleClick = () => {
  emit('click')
}
</script>

<style scoped>
.button {
  background-color: #FFFFFF;
  height: 30px;
  color:#808080;
  padding: 0.5rem 1rem;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  text-align: left;
}
.button:hover {
  background-color:#0087DC;
  color: #FFFFFF;
}

.button.active {
  background-color: #0087DC;
  color: #FFFFFF;
}
</style>
