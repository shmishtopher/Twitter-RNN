import { default as tf } from '@tensorflow/tfjs-node-gpu'
import { argv } from 'process'


const beamWidth = argv.find((_, i, args) => args[i - 1] === '--beam-width')
const beamDepth = argv.find((_, i, args) => args[i - 1] === '--beam-depth')
const seed = argv.find((_, i, args) => args[i - 1] === '--seed')


const model = tf.loadLayersModel('file://saves/model.json').then(async model => {
  console.log(await beamSearch({text: seed, prob: 0}, beamWidth, beamDepth, model))
})


async function beamSearch(seed, width, depth, model) {
  if (depth === 0) {
    return seed
  }
  else {
    const tensor = tf.oneHot(Array.from(seed.text.slice(-60), x => x.charCodeAt()), 256)
    const prediction = tf.tidy(() => model.predict(tensor.reshape([1, 60, 256])).squeeze())
    const { values, indices } = tf.topk(prediction, width)
    
    const zipped = Promise.all([indices.array(), values.array()]).then(zip)
    const branches = []
    
    for (const [ascii, prob] of await zipped) {
      branches.push({
        text: seed.text + String.fromCharCode(ascii),
        prob: seed.prob - Math.log(prob),
      })
    }
    
    tensor.dispose()
    prediction.dispose()
    values.dispose()
    indices.dispose()

    return Promise.all(branches.map(seed => beamSearch(seed, width, depth - 1, model))).then(x => x.flat())
  }
}


function* zip([lhs, rhs]) {
  const leftIterator = lhs.values()
  const rightIterator = rhs.values()
  
  let leftElement = leftIterator.next()
  let rightElement = rightIterator.next()
  
  while (!leftElement.done && !rightElement.done) {
    yield [leftElement.value, rightElement.value]
    leftElement = leftIterator.next()
    rightElement = rightIterator.next()
  }
}