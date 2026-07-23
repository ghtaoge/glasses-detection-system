import { createRouter, createWebHistory } from 'vue-router'

import AnnotationView from './views/AnnotationView.vue'
import DatasetsView from './views/DatasetsView.vue'
import ModelsView from './views/ModelsView.vue'
import HistoryView from './views/HistoryView.vue'
import OverviewView from './views/OverviewView.vue'
import RecognitionView from './views/RecognitionView.vue'
import TrainingView from './views/TrainingView.vue'

export const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', redirect: '/overview' },
    { path: '/overview', component: OverviewView },
    { path: '/datasets', component: DatasetsView },
    { path: '/annotations', component: AnnotationView },
    { path: '/training', component: TrainingView },
    { path: '/models', component: ModelsView },
    { path: '/recognition', component: RecognitionView },
    { path: '/history', component: HistoryView }
  ]
})
