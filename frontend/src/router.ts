import { createRouter, createWebHistory } from 'vue-router'

import AnnotationView from './views/AnnotationView.vue'
import DatasetsView from './views/DatasetsView.vue'
import PlaceholderView from './views/PlaceholderView.vue'

export const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', redirect: '/overview' },
    { path: '/overview', component: PlaceholderView, props: { title: '概览' } },
    { path: '/datasets', component: DatasetsView },
    { path: '/annotations', component: AnnotationView },
    { path: '/training', component: PlaceholderView, props: { title: '训练评估' } },
    { path: '/models', component: PlaceholderView, props: { title: '模型库' } },
    { path: '/recognition', component: PlaceholderView, props: { title: '识别中心' } },
    { path: '/history', component: PlaceholderView, props: { title: '历史记录' } }
  ]
})
