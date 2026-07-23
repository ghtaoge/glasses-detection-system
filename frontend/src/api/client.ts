import type { Annotation, Dataset, DatasetVersion, ImageRecord, PublicationCheck } from './types'

const API = import.meta.env.VITE_API_BASE ?? ''

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API}${path}`, init)
  if (!response.ok) {
    const body = await response.json().catch(() => null)
    throw new Error(body?.error?.message ?? `请求失败 (${response.status})`)
  }
  return response.json() as Promise<T>
}

export const api = {
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
  fetchResource: (name: string) => request(`/api/resources/${name}/fetch`, { method: 'POST' })
}
