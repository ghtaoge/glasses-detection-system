<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { Play, RefreshCw, Square } from '@lucide/vue'
import { api } from '../api/client'
import type { Dataset, DatasetVersion, EpochMetrics, TrainingSettings, TrainingTask } from '../api/types'
import { calculateTrainingProgress } from '../domain/training'

type VersionOption = DatasetVersion & { datasetName: string }
const datasets = ref<Dataset[]>([]); const versions = ref<VersionOption[]>([]); const tasks = ref<TrainingTask[]>([])
const selectedVersion = ref(''); const current = ref<TrainingTask | null>(null); const epochs = ref<EpochMetrics[]>([])
const busy = ref(false); const error = ref(''); let events: EventSource | null = null; let poll: number | null = null
const settings = ref<TrainingSettings>({ preset:'quick', epochs:5, image_size:416, batch_size:4, patience:3, device:'auto' })
const progress = computed(() => calculateTrainingProgress(current.value, epochs.value))
const active = computed(() => current.value && ['queued','running','cancelling'].includes(current.value.state))
const stateText: Record<string,string> = {queued:'等待启动',running:'训练中',cancelling:'正在取消',completed:'训练完成',failed:'训练失败',interrupted:'训练中断'}

watch(() => settings.value.preset, preset => {
  settings.value = preset === 'quick'
    ? { preset, epochs:5, image_size:416, batch_size:4, patience:3, device:settings.value.device }
    : { preset, epochs:80, image_size:640, batch_size:16, patience:15, device:settings.value.device }
})
async function load() {
  datasets.value=await api.datasets(); versions.value=[]
  for (const dataset of datasets.value) for (const version of await api.versions(dataset.id)) versions.value.push({...version,datasetName:dataset.name})
  if (!selectedVersion.value && versions.value[0]) selectedVersion.value=versions.value[0].id
  tasks.value=await api.trainingTasks(); const running=tasks.value.find(t=>['queued','running','cancelling'].includes(t.state))
  // 没有活动任务时仍恢复最近一次任务，并通过 SSE 从 sequence=0 重放历史指标。
  const visibleTask=running||tasks.value[0]
  if (visibleTask) { current.value=visibleTask; epochs.value=[]; connect(visibleTask.id) }
}
function connect(id:string) {
  // SSE 提供低延迟 epoch 指标；轮询只承担终态恢复。浏览器休眠或代理中断 SSE 时，
  // 页面仍能在下一次轮询中得到 completed/failed，避免永远停留在“训练中”。
  events?.close(); events=new EventSource(api.trainingEventsUrl(id))
  // 重连可能重放最后一条事件，按 epoch 去重后再追加到图表数据。
  events.addEventListener('epoch', event => { const item=JSON.parse((event as MessageEvent).data) as EpochMetrics; if(!epochs.value.some(e=>e.epoch===item.epoch)) epochs.value.push(item) })
  const finish=async()=>{ current.value=await api.trainingTask(id); events?.close(); if(poll) window.clearInterval(poll); tasks.value=await api.trainingTasks() }
  // 只有收到流内的终态事件后才关闭连接。若轮询抢先看到 completed，SSE 仍继续
  // 派发排在 completed 之前的所有 epoch，避免最后一批图表数据被截断。
  for (const name of ['completed','failed']) events.addEventListener(name, finish)
  const refreshState=async()=>{ current.value=await api.trainingTask(id); if(!['queued','running','cancelling'].includes(current.value.state)){ if(poll) window.clearInterval(poll); tasks.value=await api.trainingTasks() } }
  if(poll) window.clearInterval(poll); poll=window.setInterval(()=>refreshState().catch(()=>{}),1500)
}
async function start() { if(!selectedVersion.value)return; busy.value=true; error.value=''; epochs.value=[]; try { current.value=await api.startTraining(selectedVersion.value,settings.value); tasks.value.unshift(current.value); connect(current.value.id) } catch(e){error.value=(e as Error).message} finally{busy.value=false} }
async function cancel(){if(!current.value)return; current.value=await api.cancelTraining(current.value.id)}
onMounted(()=>load().catch(e=>error.value=e.message)); onBeforeUnmount(()=>{events?.close(); if(poll)window.clearInterval(poll)})
</script>

<template>
  <section><header class="page-header"><div><p class="eyebrow">模型构建</p><h1>训练评估</h1></div><button class="icon-button" title="刷新" aria-label="刷新" @click="load"><RefreshCw :size="18" /></button></header>
    <p v-if="error" class="error-banner">{{ error }}</p>
    <div class="training-layout">
      <form class="settings-panel" @submit.prevent="start"><h2>训练配置</h2><label>数据版本<select v-model="selectedVersion" :disabled="!!active"><option value="">选择已发布版本</option><option v-for="v in versions" :key="v.id" :value="v.id">{{ v.datasetName }} · 版本 {{ v.number }}</option></select></label>
        <label>预设<div class="preset-control"><button type="button" :class="{active:settings.preset==='quick'}" @click="settings.preset='quick'">快速试跑</button><button type="button" :class="{active:settings.preset==='standard'}" @click="settings.preset='standard'">标准训练</button></div></label>
        <div class="field-grid"><label>训练轮次<input v-model.number="settings.epochs" type="number" min="1" max="300" /></label><label>图像尺寸<select v-model.number="settings.image_size"><option v-for="n in [320,416,512,640,768]" :key="n">{{ n }}</option></select></label><label>批量大小<input v-model.number="settings.batch_size" type="number" min="1" max="128" /></label><label>早停轮次<input v-model.number="settings.patience" type="number" min="0" max="50" /></label></div>
        <label>计算设备<select v-model="settings.device"><option value="auto">自动选择</option><option value="cpu">CPU</option><option value="cuda">CUDA</option></select></label>
        <button v-if="!active" class="command-button primary wide" :disabled="busy||!selectedVersion" type="submit"><Play :size="17" />开始训练</button><button v-else class="command-button wide" type="button" :disabled="current?.state==='cancelling'" @click="cancel"><Square :size="15" />取消训练</button>
      </form>
      <div class="training-monitor"><div class="section-heading"><h2>当前任务</h2><span v-if="current" class="state-badge" :class="current.state">{{ stateText[current.state] }}</span></div><div v-if="current" class="monitor-body"><div class="progress-track"><span :style="{width:`${progress}%`}"></span></div><div class="metric-strip four"><div><span>进度</span><strong>{{ progress }}%</strong></div><div><span>mAP@0.5</span><strong>{{ epochs.at(-1)?.map50?.toFixed(3) || '—' }}</strong></div><div><span>Precision</span><strong>{{ epochs.at(-1)?.precision?.toFixed(3) || '—' }}</strong></div><div><span>Recall</span><strong>{{ epochs.at(-1)?.recall?.toFixed(3) || '—' }}</strong></div></div><div class="metric-chart"><div v-for="item in epochs" :key="item.epoch" class="metric-bar" :style="{height:`${Math.max(item.map50*100,2)}%`}" :title="`第 ${item.epoch} 轮: ${item.map50}`"></div><span v-if="!epochs.length">等待训练指标</span></div><p v-if="current.error_message" class="error-banner">{{ current.error_message }}</p></div><div v-else class="empty-state compact"><strong>尚未开始训练</strong></div>
        <section class="work-section"><div class="section-heading"><h2>最近任务</h2></div><div v-for="task in tasks.slice(0,6)" :key="task.id" class="task-row"><code>{{ task.id.slice(0,8) }}</code><span>{{ stateText[task.state] }}</span><span>{{ task.settings.epochs }} 轮 · {{ task.settings.image_size }} px</span></div></section></div>
    </div>
  </section>
</template>
