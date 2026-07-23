<script setup lang="ts">
import { computed, ref } from 'vue'
import type { Annotation, ClassName, PixelBox } from '../api/types'

const props = defineProps<{ image: string; imageWidth: number; imageHeight: number; annotations: Annotation[]; activeClass: ClassName }>()
const emit = defineEmits<{ create: [box: PixelBox]; select: [index: number] }>()
const stage = ref<HTMLElement | null>(null)
const start = ref<{ x: number; y: number } | null>(null)
const draft = ref<PixelBox | null>(null)

const scale = computed(() => {
  const rect = stage.value?.getBoundingClientRect()
  return { x: props.imageWidth / (rect?.width || props.imageWidth), y: props.imageHeight / (rect?.height || props.imageHeight) }
})

function point(event: PointerEvent) {
  const rect = stage.value!.getBoundingClientRect()
  return { x: (event.clientX - rect.left) * scale.value.x, y: (event.clientY - rect.top) * scale.value.y }
}
function down(event: PointerEvent) { start.value = point(event); (event.currentTarget as HTMLElement).setPointerCapture(event.pointerId) }
function move(event: PointerEvent) {
  if (!start.value) return
  const current = point(event)
  draft.value = { x1: Math.min(start.value.x,current.x), y1: Math.min(start.value.y,current.y), x2: Math.max(start.value.x,current.x), y2: Math.max(start.value.y,current.y) }
}
function up() {
  if (draft.value && draft.value.x2 - draft.value.x1 > 4 && draft.value.y2 - draft.value.y1 > 4) emit('create', draft.value)
  start.value = null; draft.value = null
}
function style(box: PixelBox) {
  return { left:`${box.x1/props.imageWidth*100}%`, top:`${box.y1/props.imageHeight*100}%`, width:`${(box.x2-box.x1)/props.imageWidth*100}%`, height:`${(box.y2-box.y1)/props.imageHeight*100}%` }
}
</script>

<template>
  <div ref="stage" class="annotation-stage" :style="{ aspectRatio: `${imageWidth}/${imageHeight}` }" @pointerdown="down" @pointermove="move" @pointerup="up">
    <img :src="image" alt="待标注图片" draggable="false" />
    <button v-for="(annotation,index) in annotations" :key="annotation.id || index" class="annotation-box" :class="annotation.class_name" :style="style(annotation.box)" @pointerdown.stop @click.stop="emit('select',index)" :aria-label="`选择标注 ${index+1}`"><span>{{ index + 1 }}</span></button>
    <div v-if="draft" class="annotation-box draft" :style="style(draft)"></div>
  </div>
</template>
