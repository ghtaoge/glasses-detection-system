<script setup lang="ts">defineProps<{ title: string }>()</script>
<template><section><header class="page-header"><div><p class="eyebrow">镜鉴工作台</p><h1>{{ title }}</h1></div></header><div class="empty-state"><Glasses :size="32" /><strong>等待数据</strong></div></section></template>
<script lang="ts">import { Glasses } from '@lucide/vue'</script>
