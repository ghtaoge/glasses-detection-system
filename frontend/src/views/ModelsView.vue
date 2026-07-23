<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { CheckCircle2, Download, RefreshCw, ShieldAlert } from '@lucide/vue'
import { api } from '../api/client'
import type { ClassName, ModelRecord } from '../api/types'
const models=ref<ModelRecord[]>([]); const busy=ref(''); const error=ref('')
const labels:Record<ClassName,string>={no_glasses:'未戴眼镜',eyeglasses:'普通眼镜',sunglasses:'墨镜'}
const isSimulated=(model:ModelRecord)=>model.metrics.simulated===true
async function load(){models.value=await api.models()}
async function activate(id:string){busy.value=id; error.value=''; try{await api.activateModel(id); await load()}catch(e){error.value=(e as Error).message}finally{busy.value=''}}
onMounted(()=>load().catch(e=>error.value=e.message))
</script>
<template><section><header class="page-header"><div><p class="eyebrow">推理资产</p><h1>模型库</h1></div><button class="icon-button" title="刷新" aria-label="刷新" @click="load"><RefreshCw :size="18" /></button></header><p v-if="error" class="error-banner">{{ error }}</p>
  <div v-if="models.length" class="model-list"><article v-for="model in models" :key="model.id" class="model-row"><div class="model-identity"><div class="model-icon" :class="isSimulated(model)?'simulated':model.quality_status"><ShieldAlert v-if="isSimulated(model)||model.quality_status!=='passed'" :size="20"/><CheckCircle2 v-else :size="20"/></div><div><h2>{{ model.name }}</h2><code>{{ model.onnx_sha256.slice(0,16) }}</code></div></div><div class="model-metrics"><div><span>mAP@0.5</span><strong>{{ model.metrics.map50.toFixed(3) }}</strong></div><div><span>mAP@0.5:0.95</span><strong>{{ model.metrics.map50_95.toFixed(3) }}</strong></div><div><span>Precision</span><strong>{{ model.metrics.precision.toFixed(3) }}</strong></div><div><span>Recall</span><strong>{{ model.metrics.recall.toFixed(3) }}</strong></div></div><div class="class-results"><span v-for="name in model.class_names" :key="name"><i :class="name"></i>{{ labels[name] }} {{ model.metrics.per_class[name]?.map50?.toFixed(2) || '—' }}</span></div><div class="model-actions"><span v-if="isSimulated(model)" class="state-badge simulated">模拟测试模型 · 不可识别</span><span v-else class="state-badge" :class="model.quality_status">{{ model.quality_status==='passed'?'达到质量门槛':'低于质量门槛' }}</span><span v-if="model.is_active&&!isSimulated(model)" class="state-badge active">当前模型</span><button v-else-if="!model.is_active" class="command-button" :title="isSimulated(model)?'模拟测试模型不能用于真实识别':''" :disabled="busy===model.id||model.quality_status!=='passed'||isSimulated(model)" @click="activate(model.id)">启用</button><a class="icon-button" :href="api.modelDownloadUrl(model.id)" title="下载 ONNX" aria-label="下载 ONNX"><Download :size="17"/></a></div></article></div><div v-else class="empty-state"><strong>训练完成后，模型会显示在这里</strong></div>
</section></template>
<style scoped>
.state-badge.simulated,.model-icon.simulated{background:#f8e8e4;color:#9b392a}
</style>
