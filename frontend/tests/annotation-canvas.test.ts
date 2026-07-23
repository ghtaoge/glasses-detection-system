import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import AnnotationCanvas from '../src/components/AnnotationCanvas.vue'

describe('AnnotationCanvas', () => {
  it('renders supplied boxes', () => {
    const wrapper = mount(AnnotationCanvas, { props: { image:'/face.jpg', imageWidth:1000, imageHeight:500, activeClass:'eyeglasses', annotations:[{ class_name:'eyeglasses', source:'manual', box:{x1:100,y1:50,x2:300,y2:250} }] } })
    const box = wrapper.get('.annotation-box')
    expect(box.attributes('style')).toContain('left: 10%')
    expect(box.attributes('style')).toContain('width: 20%')
  })
})
