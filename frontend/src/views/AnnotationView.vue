<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { Check, ChevronLeft, ChevronRight, Trash2 } from '@lucide/vue'
import { api } from '../api/client'
import AnnotationCanvas from '../components/AnnotationCanvas.vue'
import type { Annotation, ClassName, Dataset, ImageRecord, PixelBox } from '../api/types'

const datasets=ref<Dataset[]>([]); const datasetId=ref(''); const images=ref<ImageRecord[]>([]); const index=ref(0)
const annotations=ref<Annotation[]>([]); const activeClass=ref<ClassName>('no_glasses'); const selected=ref<number|null>(null); const error=ref(''); const busy=ref(false)
const current=computed(() => images.value[index.value])
async function load() { datasets.value=await api.datasets(); if(!datasetId.value && datasets.value[0]) datasetId.value=datasets.value[0].id; await loadImages() }
async function loadImages() { if(!datasetId.value) return; images.value=await api.images(datasetId.value); index.value=Math.min(index.value,Math.max(0,images.value.length-1)); sync() }
function sync() { annotations.value=(current.value?.annotations || []).map(item=>({...item,box:{...item.box}})); activeClass.value=current.value?.imported_class || 'no_glasses'; selected.value=null }
function add(box:PixelBox) { annotations.value.push({class_name:activeClass.value,box,source:'manual'}); selected.value=annotations.value.length-1 }
function remove(i:number) { annotations.value.splice(i,1); selected.value=null }
async function save(next=true) { if(!current.value) return; busy.value=true; try { await api.saveAnnotations(current.value.id,annotations.value); await loadImages(); if(next && index.value<images.value.length-1){index.value++;sync()} } catch(e){error.value=(e as Error).message} finally{busy.value=false} }
function move(delta:number){index.value=Math.max(0,Math.min(images.value.length-1,index.value+delta));sync()}
onMounted(()=>load().catch(e=>error.value=e.message))
</script>

<template>
  <section class="annotation-page">
    <header class="page-header"><div><p class="eyebrow">数据准备</p><h1>标注工作台</h1></div><select v-model="datasetId" aria-label="数据集" @change="loadImages"><option v-for="item in datasets" :key="item.id" :value="item.id">{{ item.name }}</option></select></header>
    <p v-if="error" class="error-banner">{{ error }}</p>
    <div v-if="current" class="annotation-layout">
      <aside class="image-queue"><button v-for="(image,i) in images" :key="image.id" :class="{active:i===index}" @click="index=i;sync()"><img :src="api.fileUrl(image.url)" alt="" /><span>{{ image.original_name }}</span><small>{{ image.review_state==='reviewed'?'已复核':'待复核' }}</small></button></aside>
      <div class="canvas-column"><div class="canvas-toolbar"><button class="icon-button" title="上一张" aria-label="上一张" :disabled="index===0" @click="move(-1)"><ChevronLeft :size="18" /></button><span>{{ index+1 }} / {{ images.length }}</span><button class="icon-button" title="下一张" aria-label="下一张" :disabled="index===images.length-1" @click="move(1)"><ChevronRight :size="18" /></button></div><AnnotationCanvas :image="api.fileUrl(current.url)" :image-width="current.width" :image-height="current.height" :annotations="annotations" :active-class="activeClass" @create="add" @select="selected=$event" /></div>
      <aside class="annotation-panel"><div class="section-heading"><h2>类别</h2></div><div class="segmented"><button v-for="item in [{key:'no_glasses',label:'未戴眼镜'},{key:'eyeglasses',label:'普通眼镜'},{key:'sunglasses',label:'墨镜'}]" :key="item.key" :class="{active:activeClass===item.key}" @click="activeClass=item.key as ClassName">{{ item.label }}</button></div><div class="section-heading"><h2>标注框</h2><span>{{ annotations.length }}</span></div><div class="annotation-list"><div v-for="(item,i) in annotations" :key="i" :class="{active:selected===i}" @click="selected=i"><span class="class-swatch" :class="item.class_name"></span><select v-model="item.class_name" aria-label="标注类别"><option value="no_glasses">未戴眼镜</option><option value="eyeglasses">普通眼镜</option><option value="sunglasses">墨镜</option></select><button class="icon-button danger" title="删除" aria-label="删除标注" @click.stop="remove(i)"><Trash2 :size="16" /></button></div></div><button class="command-button primary wide" :disabled="busy" @click="save(true)"><Check :size="17" />确认并下一张</button></aside>
    </div>
    <div v-else class="empty-state"><strong>暂无待标注图片</strong></div>
  </section>
</template>
