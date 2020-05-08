/**
 * @author  Shmish - c.schmitt@my.ccsu.edu
 * @version 1.0.0 - 4/24/2020
 * @license MIT - (c) Christopher K. Schmitt
 */


import { default as tf } from '@tensorflow/tfjs-node-gpu'
import { argv } from 'process'


// Read in the command line args
const beamWidth = argv.find((_, i, args) => args[i - 1] === '--beam-width')
const beamDepth = argv.find((_, i, args) => args[i - 1] === '--beam-depth')
const seed = argv.find((_, i, args) => args[i - 1] === '--seed')


// Load the model from the save file and run beam-search
// on with the given parameters.
tf.loadLayersModel('file://saves/model.json').then(async model => {
  console.log(await beamSearch({text: seed, prob: 0}, beamWidth, beamDepth, model))
})


/**
 * BeamSearch is a recusave tree search implentation which
 * keeps track of the top "width" probabilities at each
 * level.  Ruturns a promise containing each of the final
 * token sequences.
 * 
 * @param {String} seed 
 * @param {Number} width 
 * @param {Number} depth 
 * @param {tf.Sequential} model 
 */
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


/**
 * zip is a utility function that produces an iterable of
 * two arrays zipped together.
 * 
 * @param {Array<Array<any>>} param0 - Collection of arrays to zip
 */
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