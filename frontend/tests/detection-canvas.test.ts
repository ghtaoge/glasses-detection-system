import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import DetectionCanvas from '../src/components/DetectionCanvas.vue'

describe('DetectionCanvas', () => {
  it('scales original-pixel detections as percentages', () => {
    const wrapper = mount(DetectionCanvas, {
      props: {
        src: '/result.jpg', width: 1000, height: 500,
        detections: [{ class_name: 'eyeglasses', confidence: .9, box: { x1: 100, y1: 50, x2: 300, y2: 250 } }]
      }
    })

    const box = wrapper.get('.detection-box')
    expect(box.attributes('style')).toContain('left: 10%')
    expect(box.attributes('style')).toContain('width: 20%')
    expect(wrapper.text()).toContain('普通眼镜 90%')
  })
})
