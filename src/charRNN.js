import { default as tf } from '@tensorflow/tfjs-node'


/**
 * createRNN creates and compiles the RNN and returns
 * the result.  This network is constructed with a 
 * standard GRU layer followed by a multi-layered
 * perceptron augmented with dropout layers.
 */
export function createRNN() {
  const model = tf.sequential()

  model.add(tf.layers.gru({
    batchInputShape: [512, 60, 256],
    recurrentInitializer: 'glorotNormal',
    units: 1024,
    stateful: true,
  }))
  
  model.add(tf.layers.dense({
    units: 512,
    activation: 'relu'
  }))
  
  model.add(tf.layers.dropout({ rate: 0.1 }))
  
  model.add(tf.layers.dense({
    units: 512,
    activation: 'relu'
  }))
  
  model.add(tf.layers.dropout({ rate: 0.1 }))
  
  model.add(tf.layers.dense({
    units: 512,
    activation: 'relu'
  }))
  
  model.add(tf.layers.dense({
    units: 256,
    activation: 'softmax',
  }))

  model.compile({
    optimizer: 'adam',
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  })

  return model
}