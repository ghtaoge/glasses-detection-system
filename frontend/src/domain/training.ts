import type { EpochMetrics, TrainingTask } from '../api/types'

/**
 * 将任务状态和最近一次 epoch 合并为界面进度。
 *
 * 训练可能因为 early stopping 提前结束；只要后端已经确认 completed，工作流就是
 * 完整结束，因此显示 100%。运行中仍按实际 epoch 计算，并限制在 0-100 范围内。
 */
export function calculateTrainingProgress(
  task: TrainingTask | null,
  epochs: EpochMetrics[]
): number {
  if (task?.state === 'completed') return 100
  const last = epochs.at(-1)
  if (!last || last.epochs <= 0) return 0
  return Math.min(100, Math.max(0, Math.round(last.epoch / last.epochs * 100)))
}
