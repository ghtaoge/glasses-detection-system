<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { CheckCircle2, Download, PackagePlus, RefreshCw, ShieldAlert } from '@lucide/vue'
import { api } from '../api/client'
import type { ClassName, ModelRecord } from '../api/types'

const models = ref<ModelRecord[]>([])
const busy = ref('')
const error = ref('')
const labels: Record<ClassName, string> = {
  no_glasses: '未戴眼镜',
  eyeglasses: '普通眼镜',
  sunglasses: '墨镜'
}
const isSimulated = (model: ModelRecord) => model.metrics.simulated === true
const isPretrained = (model: ModelRecord) => model.metrics.pretrained === true

async function load() {
  models.value = await api.models()
}

async function installPretrained() {
  busy.value = 'pretrained'
  error.value = ''
  try {
    await api.installPretrainedModel()
    await load()
  } catch (reason) {
    error.value = (reason as Error).message
  } finally {
    busy.value = ''
  }
}

async function activate(id: string) {
  busy.value = id
  error.value = ''
  try {
    await api.activateModel(id)
    await load()
  } catch (reason) {
    error.value = (reason as Error).message
  } finally {
    busy.value = ''
  }
}

onMounted(() => load().catch(reason => error.value = reason.message))
</script>

<template>
  <section>
    <header class="page-header">
      <div><p class="eyebrow">推理资产</p><h1>模型库</h1></div>
      <div class="header-actions">
        <button class="command-button compact" :disabled="busy === 'pretrained'" @click="installPretrained">
          <PackagePlus :size="17" />{{ busy === 'pretrained' ? '正在安装' : '安装公开模型' }}
        </button>
        <button class="icon-button" title="刷新" aria-label="刷新" @click="load"><RefreshCw :size="18" /></button>
      </div>
    </header>
    <p v-if="error" class="error-banner">{{ error }}</p>

    <div v-if="models.length" class="model-list">
      <article v-for="model in models" :key="model.id" class="model-row">
        <div class="model-identity">
          <div class="model-icon" :class="isSimulated(model) ? 'simulated' : model.quality_status">
            <ShieldAlert v-if="isSimulated(model) || model.quality_status === 'below_target'" :size="20" />
            <CheckCircle2 v-else :size="20" />
          </div>
          <div>
            <h2>{{ model.name }}</h2>
            <code>{{ model.onnx_sha256.slice(0, 16) }}</code>
          </div>
        </div>

        <div v-if="isPretrained(model)" class="model-metrics">
          <div><span>任意眼镜 F1</span><strong>{{ model.metrics.published_f1?.anyglasses.toFixed(3) }}</strong></div>
          <div><span>墨镜 F1</span><strong>{{ model.metrics.published_f1?.sunglasses.toFixed(3) }}</strong></div>
          <div><span>人脸检测</span><strong>YuNet</strong></div>
          <div><span>运行时</span><strong>ONNX</strong></div>
        </div>
        <div v-else class="model-metrics">
          <div><span>mAP@0.5</span><strong>{{ model.metrics.map50.toFixed(3) }}</strong></div>
          <div><span>mAP@0.5:0.95</span><strong>{{ model.metrics.map50_95.toFixed(3) }}</strong></div>
          <div><span>Precision</span><strong>{{ model.metrics.precision.toFixed(3) }}</strong></div>
          <div><span>Recall</span><strong>{{ model.metrics.recall.toFixed(3) }}</strong></div>
        </div>

        <div class="class-results">
          <span v-for="name in model.class_names" :key="name">
            <i :class="name"></i>{{ labels[name] }}
            <template v-if="!isPretrained(model)">{{ model.metrics.per_class[name]?.map50?.toFixed(2) || '—' }}</template>
          </span>
        </div>
        <div class="model-actions">
          <span v-if="isSimulated(model)" class="state-badge simulated">模拟测试模型 · 不可识别</span>
          <span v-else-if="isPretrained(model)" class="state-badge pretrained">公开预训练 · MIT</span>
          <span v-else class="state-badge" :class="model.quality_status">
            {{ model.quality_status === 'passed' ? '达到质量门槛' : '低于质量门槛' }}
          </span>
          <span v-if="model.is_active && !isSimulated(model)" class="state-badge active">当前模型</span>
          <button
            v-else-if="!model.is_active"
            class="command-button"
            :title="isSimulated(model) ? '模拟测试模型不能用于真实识别' : ''"
            :disabled="busy === model.id || model.quality_status === 'below_target' || isSimulated(model)"
            @click="activate(model.id)"
          >启用</button>
          <a class="icon-button" :href="api.modelDownloadUrl(model.id)" title="下载 ONNX" aria-label="下载 ONNX">
            <Download :size="17" />
          </a>
        </div>
      </article>
    </div>
    <div v-else class="empty-state"><strong>安装公开模型，或完成一次真实训练</strong></div>
  </section>
</template>

<style scoped>
.header-actions{display:flex;align-items:center;gap:8px}
.command-button.compact{width:auto;padding:0 13px;gap:7px}
.state-badge.simulated,.model-icon.simulated{background:#f8e8e4;color:#9b392a}
.state-badge.pretrained,.model-icon.pretrained{background:#e4eef8;color:#245e96}
</style>
