import { describe, expect, it } from 'vitest'
import type { EpochMetrics, TrainingTask } from '../src/api/types'
import { calculateTrainingProgress } from '../src/domain/training'

function task(state: TrainingTask['state']): TrainingTask {
  return {
    id: 'task', dataset_version_id: 'version', state,
    settings: { preset: 'standard', epochs: 80, image_size: 640, batch_size: 16, patience: 15, device: 'cpu' },
    result: null, error_code: null, error_message: null,
    created_at: '2026-07-23T00:00:00Z', updated_at: '2026-07-23T00:00:00Z'
  }
}

function epoch(current: number): EpochMetrics {
  return { epoch: current, epochs: 80, loss: .2, precision: .8, recall: .8, map50: .8, map50_95: .6 }
}

describe('calculateTrainingProgress', () => {
  it('reports actual epoch progress while running', () => {
    expect(calculateTrainingProgress(task('running'), [epoch(71)])).toBe(89)
  })

  it('reports 100 after successful completion even when training stopped early', () => {
    expect(calculateTrainingProgress(task('completed'), [epoch(71)])).toBe(100)
  })
})
