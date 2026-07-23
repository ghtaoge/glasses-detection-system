<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { CheckCircle2, Database, Plus, RefreshCw, Upload } from '@lucide/vue'
import { api } from '../api/client'
import type { ClassName, Dataset, DatasetVersion, PublicationCheck } from '../api/types'

const datasets = ref<Dataset[]>([]); const selectedId = ref(''); const versions = ref<DatasetVersion[]>([])
const check = ref<PublicationCheck | null>(null); const error = ref(''); const busy = ref(false); const name = ref('')
const uploadClass = ref<ClassName>('no_glasses'); const fileInput = ref<HTMLInputElement | null>(null)
const selected = computed(() => datasets.value.find(item => item.id === selectedId.value))

async function load() { datasets.value = await api.datasets(); if (!selectedId.value && datasets.value[0]) selectedId.value = datasets.value[0].id; await loadSelected() }
async function loadSelected() { if (!selectedId.value) return; [versions.value,check.value] = await Promise.all([api.versions(selectedId.value),api.publicationCheck(selectedId.value)]) }
async function create() { if (!name.value.trim()) return; busy.value=true; try { const item=await api.createDataset(name.value); selectedId.value=item.id; name.value=''; await load() } catch(e) { error.value=(e as Error).message } finally { busy.value=false } }
async function upload() { const files=Array.from(fileInput.value?.files || []); if (!files.length || !selectedId.value) return; busy.value=true; try { await api.uploadImages(selectedId.value,files,uploadClass.value); if(fileInput.value) fileInput.value.value=''; await load() } catch(e) { error.value=(e as Error).message } finally { busy.value=false } }
async function publish() { if (!selectedId.value) return; busy.value=true; try { await api.publish(selectedId.value); await loadSelected() } catch(e) { error.value=(e as Error).message } finally { busy.value=false } }
onMounted(() => load().catch(e => error.value=e.message))
</script>

<template>
  <section>
    <header class="page-header"><div><p class="eyebrow">数据准备</p><h1>数据集</h1></div><button class="icon-button" title="刷新" aria-label="刷新" @click="load"><RefreshCw :size="18" /></button></header>
    <p v-if="error" class="error-banner">{{ error }}</p>
    <div class="dataset-layout">
      <aside class="dataset-list">
        <form class="inline-form" @submit.prevent="create"><input v-model="name" aria-label="数据集名称" placeholder="数据集名称" /><button class="icon-button primary" aria-label="新建数据集" title="新建数据集" :disabled="busy"><Plus :size="18" /></button></form>
        <button v-for="item in datasets" :key="item.id" class="dataset-item" :class="{active:item.id===selectedId}" @click="selectedId=item.id; loadSelected()"><Database :size="17" /><span><strong>{{ item.name }}</strong><small>{{ item.image_count }} 张 · {{ item.pending_count }} 待复核</small></span></button>
      </aside>
      <div v-if="selected" class="dataset-workspace">
        <div class="metric-strip"><div><span>图片</span><strong>{{ selected.image_count }}</strong></div><div><span>待复核</span><strong>{{ selected.pending_count }}</strong></div><div><span>版本</span><strong>{{ versions.length }}</strong></div></div>
        <section class="work-section"><div class="section-heading"><h2>导入图片</h2></div><div class="upload-row"><select v-model="uploadClass" aria-label="导入类别"><option value="no_glasses">未戴眼镜</option><option value="eyeglasses">普通眼镜</option><option value="sunglasses">墨镜</option></select><input ref="fileInput" type="file" accept="image/jpeg,image/png,image/webp" multiple aria-label="选择图片" /><button class="command-button" :disabled="busy" @click="upload"><Upload :size="17" />导入</button></div></section>
        <section class="work-section"><div class="section-heading"><h2>发布检查</h2><span v-if="check?.ready" class="status success"><CheckCircle2 :size="15" />可以发布</span></div><div class="class-grid"><div v-for="(label,key) in {no_glasses:'未戴眼镜',eyeglasses:'普通眼镜',sunglasses:'墨镜'}" :key="key"><span>{{ label }}</span><strong>{{ check?.class_counts[key as ClassName] || 0 }}</strong></div></div><ul v-if="check?.errors.length" class="issue-list"><li v-for="item in check.errors" :key="item.code">{{ item.message }}</li></ul><button class="command-button primary" :disabled="busy || !check?.ready" @click="publish">发布版本</button></section>
        <section class="work-section"><div class="section-heading"><h2>数据版本</h2></div><div v-if="!versions.length" class="empty-inline">暂无已发布版本</div><div v-for="version in versions" :key="version.id" class="version-row"><strong>版本 {{ version.number }}</strong><span>{{ version.split_counts.train || 0 }} / {{ version.split_counts.val || 0 }} / {{ version.split_counts.test || 0 }}</span><code>{{ version.manifest_sha256.slice(0,12) }}</code></div></section>
      </div>
      <div v-else class="empty-state"><Database :size="30" /><strong>新建一个数据集</strong></div>
    </div>
  </section>
</template>
