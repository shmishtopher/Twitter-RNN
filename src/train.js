/**
 * @author  Shmish - c.schmitt@my.ccsu.edu
 * @version 1.0.0 - 4/24/2020
 * @license MIT - (c) Christopher K. Schmitt
 */


import { default as dataset } from './data.js'
import { createRNN } from './charRNN.js'


const model = createRNN()


// Train the model on the dataset in a streaming manner
// Save at the end of each epoch.
model.fitDataset(dataset, {
  epochs: 200,
  verbos: 2,
  callbacks: {
    onEpochEnd(epoch) { model.save(`file://saves/${epoch}`) },
  }
})
