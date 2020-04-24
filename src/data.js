/**
 * @author  Shmish - c.schmitt@my.ccsu.edu
 * @version 1.0.0 - 4/24/2020
 * @license MIT - (c) Christopher K. Schmitt
 */


import { createReadStream } from 'fs'
import { createInterface } from 'readline'
import { default as tf } from '@tensorflow/tfjs-node'


/**
 * csvParse provides an interface for parsing CSV files.
 * If the `headers` field is provided then the generator
 * will be index by key name instead.
 * 
 * @param {string} file - The location of the file to parse
 * @param {string[]} headers - The headers of the csv file
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
 * Zip takes as argument two arrays and zips them
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