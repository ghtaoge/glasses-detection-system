<script setup lang="ts">
import { computed, onBeforeUnmount, ref } from 'vue'
import { Camera, ImageUp, LoaderCircle, ScanSearch } from '@lucide/vue'
import { api } from '../api/client'
import type { ClassName, DetectionRecord } from '../api/types'
import DetectionCanvas from '../components/DetectionCanvas.vue'
import CameraStage from '../components/CameraStage.vue'
const tab=ref<'image'|'camera'>('image'); const file=ref<File|null>(null); const preview=ref(''); const result=ref<DetectionRecord|null>(null)
const confidence=ref(.25); const busy=ref(false); const error=ref('')
const snapshotSaved=ref(false)
const labels:Record<ClassName,string>={no_glasses:'未戴眼镜',eyeglasses:'普通眼镜',sunglasses:'墨镜'}
const resultUrl=computed(()=>result.value?api.fileUrl(result.value.original_url):preview.value)
function select(event:Event){const selected=(event.target as HTMLInputElement).files?.[0]; if(!selected)return; if(preview.value)URL.revokeObjectURL(preview.value); file.value=selected; preview.value=URL.createObjectURL(selected); result.value=null; error.value=''}
async function detect(){if(!file.value)return;busy.value=true;error.value='';try{result.value=await api.detectImage(file.value,confidence.value)}catch(e){error.value=(e as Error).message}finally{busy.value=false}}
onBeforeUnmount(()=>{if(preview.value)URL.revokeObjectURL(preview.value)})
</script>
<template><section><header class="page-header"><div><p class="eyebrow">本地推理</p><h1>识别中心</h1></div></header><div class="view-tabs"><button :class="{active:tab==='image'}" @click="tab='image'"><ImageUp :size="16"/>图片检测</button><button :class="{active:tab==='camera'}" @click="tab='camera'"><Camera :size="16"/>实时摄像头</button></div>
  <p v-if="error" class="error-banner">{{ error }}</p><div v-if="tab==='image'" class="recognition-layout"><div class="media-workspace"><DetectionCanvas v-if="result&&resultUrl" :src="resultUrl" :width="result.width" :height="result.height" :detections="result.detections"/><img v-else-if="preview" class="image-preview" :src="preview" alt="待检测图片"/><label v-else class="drop-target"><ImageUp :size="32"/><strong>选择一张图片</strong><input type="file" accept="image/jpeg,image/png,image/webp" aria-label="选择图片" @change="select"/></label></div><aside class="recognition-panel"><label class="file-command"><input type="file" accept="image/jpeg,image/png,image/webp" aria-label="更换图片" @change="select"/><ImageUp :size="16"/>{{ file?'更换图片':'选择图片' }}</label><label class="slider-field"><span>置信度 <strong>{{ Math.round(confidence*100) }}%</strong></span><input v-model.number="confidence" type="range" min=".05" max=".95" step=".05"/></label><button class="command-button primary wide" :disabled="!file||busy" @click="detect"><LoaderCircle v-if="busy" class="spin" :size="17"/><ScanSearch v-else :size="17"/>开始识别</button><div v-if="result" class="result-summary"><div><span>人脸</span><strong>{{ result.detections.length }}</strong></div><div><span>耗时</span><strong>{{ result.duration_ms.toFixed(1) }} ms</strong></div><p>{{ result.device }}</p><ul><li v-for="(item,index) in result.detections" :key="index"><i :class="item.class_name"></i><span>{{ labels[item.class_name] }}</span><strong>{{ Math.round(item.confidence*100) }}%</strong></li></ul></div></aside></div>
  <div v-else><p v-if="snapshotSaved" class="success-banner">快照已保存到历史记录</p><CameraStage @saved="snapshotSaved=true"/></div>
</section></template>
