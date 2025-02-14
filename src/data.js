/**
 * @author  Shmish - c.schmitt@my.ccsu.edu
 * @version 1.0.0 - 4/24/2020
 * @license MIT - (c) Christopher K. Schmitt
 */


import { createReadStream } from 'fs'
import { createInterface } from 'readline'
import { default as tf } from '@tensorflow/tfjs-node-gpu'


/**
 * csvParse provides an interface for parsing CSV files.
 * If the `headers` field is provided then the generator
 * will be index by key name instead.
 * 
 * @param {String} file - The location of the file to parse
 * @param {String[]} headers - The headers of the csv file
 */
async function* csvParse(file, headers) {
  const fileStream = createReadStream(file)
  const lineStream = createInterface(fileStream)

  for await (const line of lineStream) {
    const row = line
      .substring(1, line.length - 1)
      .trimRight()
      .split('","')
    
    if (headers !== undefined) {
      yield Object.fromEntries(zip(headers, row))
    }
    else {
      yield row
    }
  }
}


/**
 * zip takes as argument two arrays and zips them
 * together element-wise
 * 
 * @param {Array<any>} lhs - left hand array
 * @param {Array<any>} rhs - right hand array
 */
function* zip(lhs, rhs) {
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


/**
 * roll provides frames for rolling window calculations
 * 
 * @param {Iterable<any>} iteratble - the iterable to roll over
 * @param {Number} size - the size of the rolling window
 */
function* roll(iteratble, size) {
  const buffer = []

  for (const element of iteratble) {
    buffer.push(element)

    if (buffer.length === size) {
      yield buffer
      buffer.shift()
    }
  }
}


/**
 * An "all" implementation for aritrary iterables.
 * Returns true if every element of the iterable
 * returns true for the provided predicate.
 * 
 * @param {Iterable<any>} iterable 
 * @param {Function} predicate 
 */
function every(iterable, predicate) {
  for (const element of iterable) {
    if (predicate(element) !== true) {
      return false
    }
  }
  return true
}


// Constants for the datafile
const dataFile = 'data/tweets.csv'
const headers = ['label', 'id', 'date', 'query', 'user', 'tweet']


/**
 * dataset is a generator that yields tensor pairs by
 * rolling over tweets from the provided dataset.
 */
async function* dataset() {
  for await (const { tweet } of csvParse(dataFile, headers)) {
    for (const substring of roll(tweet, 61)) {
      const input = substring.slice(0, 60)
      const target = substring[60]

      yield {
        xs: tf.oneHot(Array.from(input, x => x.charCodeAt()), 256),
        ys: tf.oneHot([target.charCodeAt()], 256),
      }
    }
  }
}


// Export dataset
export default tf.data.generator(dataset)
  .take(2000)
  .batch(512)

