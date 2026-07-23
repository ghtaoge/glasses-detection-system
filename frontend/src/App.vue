<script setup lang="ts">
import { Camera, ChartNoAxesCombined, Database, Glasses, History, Images, ScanFace } from '@lucide/vue'

const navigation = [
  { to: '/overview', label: '概览', icon: ChartNoAxesCombined },
  { to: '/datasets', label: '数据集', icon: Database },
  { to: '/annotations', label: '标注工作台', icon: Images },
  { to: '/training', label: '训练评估', icon: ScanFace },
  { to: '/models', label: '模型库', icon: Glasses },
  { to: '/recognition', label: '识别中心', icon: Camera },
  { to: '/history', label: '历史记录', icon: History }
]
</script>

<template>
  <div class="app-shell">
    <aside class="sidebar">
      <div class="brand"><span class="brand-mark"><Glasses :size="22" /></span><span>镜鉴</span></div>
      <nav aria-label="主导航">
        <RouterLink v-for="item in navigation" :key="item.to" :to="item.to" class="nav-link">
          <component :is="item.icon" :size="18" /><span>{{ item.label }}</span>
        </RouterLink>
      </nav>
      <div class="local-state"><span class="status-dot"></span>本地运行</div>
    </aside>
    <main class="main-content"><RouterView /></main>
  </div>
</template>
