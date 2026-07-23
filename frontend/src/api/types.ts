export type ClassName = 'no_glasses' | 'eyeglasses' | 'sunglasses'

export interface PixelBox {
  x1: number
  y1: number
  x2: number
  y2: number
}

export interface Annotation {
  id?: string
  class_name: ClassName
  box: PixelBox
  source: 'manual' | 'yunet'
}

export interface ImageRecord {
  id: string
  dataset_id: string
  url: string
  original_name: string
  width: number
  height: number
  review_state: 'needs_review' | 'auto_annotated' | 'reviewed'
  imported_class: ClassName | null
  annotations: Annotation[]
}

export interface Dataset {
  id: string
  name: string
  image_count: number
  pending_count: number
  created_at?: string
}

export interface DatasetVersion {
  id: string
  dataset_id: string
  number: number
  split_counts: Record<string, number>
  class_counts: Record<ClassName, number>
  manifest_sha256: string
  created_at?: string
}

export interface PublicationCheck {
  ready: boolean
  errors: Array<{ code: string; message: string }>
  warnings: Array<{ code: string; message: string }>
  image_count: number
  class_counts: Partial<Record<ClassName, number>>
  pending_count: number
}
