import { createApp } from 'vue'

import App from './App.vue'
import { router } from './router'
import './styles/base.css'
import './styles/training.css'
import './styles/inference.css'
import './styles/overview-camera.css'
import './styles/labels.css'

createApp(App).use(router).mount('#app')
