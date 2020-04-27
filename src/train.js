import { default as tf } from '@tensorflow/tfjs-node'
import { default as dataset } from './data.js'
import { createRNN } from './charRNN.js'


const model = createRNN()


model.fitDataset(dataset, {
  epochs: 20,
  verbos: 2,
  callbacks: {
    onEpochEnd(epoch) {
      model.save(`file://saves/${epoch}`)
    }
  },
})