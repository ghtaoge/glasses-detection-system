import type {
  Annotation, Dataset, DatasetVersion, DetectionRecord, ImageRecord, ModelRecord, PublicationCheck,
  Overview, TrainingSettings, TrainingTask
} from './types'

const API = import.meta.env.VITE_API_BASE ?? ''
const WS_API = API ? API.replace(/^http/, 'ws') : `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}`

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API}${path}`, init)
  if (!response.ok) {
    const body = await response.json().catch(() => null)
    throw new Error(body?.error?.message ?? `请求失败 (${response.status})`)
  }
  if (response.status === 204) return undefined as T
  return response.json() as Promise<T>
}

export const api = {
  fileUrl: (path: string) => path.startsWith('/api/') ? `${API}${path}` : path,
  cameraSocketUrl: () => `${WS_API}/api/camera/ws`,
  datasets: () => request<Dataset[]>('/api/datasets'),
  createDataset: (name: string) => request<Dataset>('/api/datasets', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name })
  }),
  images: (datasetId: string, reviewState?: string) => request<ImageRecord[]>(
    `/api/datasets/${datasetId}/images${reviewState ? `?review_state=${reviewState}` : ''}`
  ),
  saveAnnotations: (imageId: string, annotations: Annotation[]) => request<ImageRecord>(
    `/api/images/${imageId}/annotations`, {
      method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(annotations)
    }
  ),
  uploadImages: (datasetId: string, files: File[], className: string | null) => {
    const body = new FormData()
    for (const file of files) body.append('files', file)
    if (className) body.append('class_name', className)
    return request<{ imported: number; duplicate: number; invalid: number }>(
      `/api/datasets/${datasetId}/imports/images`, { method: 'POST', body }
    )
  },
  publicationCheck: (datasetId: string) => request<PublicationCheck>(
    `/api/datasets/${datasetId}/publication-check`
  ),
  versions: (datasetId: string) => request<DatasetVersion[]>(`/api/datasets/${datasetId}/versions`),
  publish: (datasetId: string) => request<DatasetVersion>(`/api/datasets/${datasetId}/versions`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ seed: 20260723 })
  }),
  fetchResource: (name: string) => request(`/api/resources/${name}/fetch`, { method: 'POST' }),
  trainingTasks: () => request<TrainingTask[]>('/api/training'),
  trainingTask: (id: string) => request<TrainingTask>(`/api/training/${id}`),
  startTraining: (datasetVersionId: string, settings: TrainingSettings) => request<TrainingTask>(
    '/api/training', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ dataset_version_id: datasetVersionId, ...settings })
    }
  ),
  cancelTraining: (id: string) => request<TrainingTask>(`/api/training/${id}/cancel`, { method: 'POST' }),
  models: () => request<ModelRecord[]>('/api/models'),
  installPretrainedModel: () => request<ModelRecord>('/api/models/pretrained/install', {
    method: 'POST'
  }),
  activateModel: (id: string) => request<ModelRecord>(`/api/models/${id}/activate`, { method: 'POST' }),
  modelDownloadUrl: (id: string) => `${API}/api/models/${id}/download`,
  trainingEventsUrl: (id: string) => `${API}/api/training/${id}/events`,
  detectImage: (file: File, confidence: number) => {
    const body=new FormData(); body.append('file',file); body.append('confidence',String(confidence))
    return request<DetectionRecord>('/api/inference/image',{method:'POST',body})
  },
  saveSnapshot: (file: File, confidence: number) => {
    const body=new FormData(); body.append('frame',file); body.append('confidence',String(confidence))
    return request<DetectionRecord>('/api/camera/snapshots',{method:'POST',body})
  },
  history: (filters: {source?:string;class_name?:string}={}) => {
    const query=new URLSearchParams(Object.entries(filters).filter(([,value])=>value) as string[][])
    return request<{items:DetectionRecord[];next_cursor:null}>(`/api/history${query.size?`?${query}`:''}`)
  },
  historyRecord: (id:string) => request<DetectionRecord>(`/api/history/${id}`),
  deleteHistory: (id:string) => request<void>(`/api/history/${id}`,{method:'DELETE'}),
  overview: () => request<Overview>('/api/overview')
}
