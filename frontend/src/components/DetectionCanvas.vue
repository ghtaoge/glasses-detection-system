<script setup lang="ts">
import type { ClassName, DetectionItem } from '../api/types'
defineProps<{src:string;width:number;height:number;detections:DetectionItem[]}>()
const labels:Record<ClassName,string>={no_glasses:'未戴眼镜',eyeglasses:'普通眼镜',sunglasses:'墨镜'}
</script>
<template><div class="detection-stage" :style="{aspectRatio:`${width}/${height}`}"><img :src="src" alt="检测图片"/><div v-for="(item,index) in detections" :key="item.id||index" class="detection-box" :class="item.class_name" :style="{left:`${item.box.x1/width*100}%`,top:`${item.box.y1/height*100}%`,width:`${(item.box.x2-item.box.x1)/width*100}%`,height:`${(item.box.y2-item.box.y1)/height*100}%`}"><span>{{ labels[item.class_name] }} {{ Math.round(item.confidence*100) }}%</span></div></div></template>
