import { default as dataset } from './data.js'
import { createRNN } from './charRNN.js'


const model = createRNN()


model.fitDataset(dataset, {
  epochs: 20,
  verbos: 2,
  callbacks: {
    onBatchEnd(batch) { console.log(`${batch}/1000`) },
    onEpochEnd(epoch) { model.save(`file://saves/${epoch}`) },
  }
})
