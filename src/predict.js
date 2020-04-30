import { default as tf } from '@tensorflow/tfjs-node'


const model = tf.loadLayersModel('file://saves/2/model.json')
const tweet = "@switchfoot http://twitpic.com/2y1zl - Awwww, You shoulda got David Carr of Third Day to do it.".slice(0, 60)


model.then(model => {
  const predictor = tf.sequential()

  predictor.add(tf.layers.gru({
    inputShape: [60, 256],
    recurrentInitializer: 'glorotNormal',
    units: 1024,
  }))

  predictor.add(tf.layers.dense({
    units: 512,
    activation: 'relu'
  }))
  
  predictor.add(tf.layers.dropout({ rate: 0.1 }))
  
  predictor.add(tf.layers.dense({
    units: 512,
    activation: 'relu'
  }))
  
  predictor.add(tf.layers.dropout({ rate: 0.1 }))
  
  predictor.add(tf.layers.dense({
    units: 512,
    activation: 'relu'
  }))
  
  predictor.add(tf.layers.dense({
    units: 256,
    activation: 'softmax',
  }))

  predictor.compile({
    optimizer: 'adam',
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  })

  predictor.summary()
  predictor.setWeights(model.getWeights())

  const tensor = tf.oneHot(Array.from(tweet, x => x.charCodeAt()), 256)
  const prediction = predictor.predict(tensor.reshape([1, tweet.length, 256]))
  console.log(tweet)

  prediction.squeeze().argMax().print()
  tf.multinomial(prediction.squeeze(), 5).print()
})