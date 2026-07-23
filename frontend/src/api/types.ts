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

export type TrainingState = 'queued' | 'running' | 'cancelling' | 'completed' | 'failed' | 'interrupted'

export interface TrainingSettings {
  preset: 'quick' | 'standard'
  epochs: number
  image_size: number
  batch_size: number
  patience: number
  device: 'auto' | 'cpu' | 'cuda'
}

export interface EpochMetrics {
  epoch: number
  epochs: number
  loss: number
  precision: number
  recall: number
  map50: number
  map50_95: number
}

export interface EvaluationMetrics {
  map50: number
  map50_95: number
  precision: number
  recall: number
  per_class: Partial<Record<ClassName, { map50: number }>>
  confusion_matrix?: number[][]
  simulated?: boolean
  pretrained?: boolean
  engine?: string
  source?: string
  license?: string
  published_f1?: { anyglasses: number; sunglasses: number }
}

export interface TrainingTask {
  id: string
  dataset_version_id: string
  state: TrainingState
  settings: TrainingSettings
  result: { model_id?: string; metrics?: EvaluationMetrics } | null
  error_code: string | null
  error_message: string | null
  created_at: string
  updated_at: string
}

export interface ModelRecord {
  id: string
  training_task_id: string | null
  name: string
  onnx_sha256: string
  class_names: ClassName[]
  metrics: EvaluationMetrics
  quality_status: 'passed' | 'below_target' | 'pretrained'
  is_active: boolean
  created_at: string
}

export interface DetectionItem {
  id?: string
  class_name: ClassName
  confidence: number
  box: PixelBox
}

export interface DetectionRecord {
  id: string
  source: 'image' | 'camera'
  model_id: string
  original_url: string
  annotated_url: string
  width: number
  height: number
  duration_ms: number
  device: string
  delete_state: 'active' | 'pending'
  created_at: string
  detections: DetectionItem[]
}

export interface Overview {
  datasets: { count: number; image_count: number; pending_count: number }
  active_model: ModelRecord | null
  latest_training: TrainingTask | null
  latest_evaluation: EvaluationMetrics | null
  detection_counts: { total: number; image: number; camera: number }
  storage_bytes: number
  resources: Array<{name:string;filename:string;license:string;ready:boolean}>
}
