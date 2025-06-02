// const tf = require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const readline = require('readline');

// Settings
const INPUT_SIZE = 6;
const NUM_CLASSES = 3;

// Helper functions
function normalize(x) {
  return x; // 4.5579 * Math.log(x); // Adjust if your range is different
}

function classify(x) {
  if (x <= 2) return 0;
  if (x <= 10) return 1;
  return 2;
}

function oneHot(index) {
  const arr = [0, 0, 0];
  arr[index] = 1;
  return arr;
}

// Generate training data
function prepareData(sequences) {
  const inputs = [];
  const labels = [];

  sequences.forEach(seq => {
    for (let i = 0; i < seq.length - INPUT_SIZE - 1; i++) {
      const input = seq.slice(i, i + INPUT_SIZE).map(normalize);
      const label = classify(seq[i + INPUT_SIZE]);
      inputs.push(input);
      labels.push(oneHot(label));
    }
  });

  return {
    xs: tf.tensor2d(inputs),
    ys: tf.tensor2d(labels)
  };
}

// Model definition
function buildModel() {
  const input = tf.input({ shape: [INPUT_SIZE] });

  const x = tf.layers.dense({ units: 64, activation: 'relu' }).apply(input);
  const x2 = tf.layers.dropout({ rate: 0.2 }).apply(x);
  const x3 = tf.layers.dense({ units: 64, activation: 'relu' }).apply(x2);
  const output = tf.layers.dense({ units: NUM_CLASSES, activation: 'softmax' }).apply(x3);

  const model = tf.model({ inputs: input, outputs: output });

  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });

  return model;
}

// Example training sequences
const sequences = require('./data.js');
// [
//   [1.1, 1.2, 1.3, 1.4, 1.5, 2.1, 3.5, 4.5, 10.2, 11.3, 1.8, 2.0, 20.0],
//   [2.2, 2.3, 8.0, 1.1, 2.1, 1.9, 10.5, 11.1, 2.2, 3.0, 2.5, 2.6, 9.9],
//   [1.0, 1.1, 1.3, 1.5, 1.8, 2.3, 5.0, 6.0, 12.0, 2.2, 1.9, 1.8, 2.1],
// ];

// Train the model
async function trainAndPredict(testInput) {
  const { xs, ys } = prepareData(sequences);
  const model = buildModel();

  // console.log("Training...");
  await model.fit(xs, ys, {
    epochs: 40,
    batchSize: 8,
    shuffle: true,
    callbacks: {
      onEpochEnd: (epoch, logs) =>
        { 
          //console.log(`Epoch ${epoch + 1}: Loss = ${logs.loss.toFixed(4)}, Accuracy = ${(logs.acc * 100).toFixed(2)}%`) 
        }
    }
  });

  // Make prediction
  // const testInput = [7.39, 3.39, 44.54, 1.87, 1.58, 3.25];
  
  const normTestInput = tf.tensor2d([testInput.map(normalize)]);

  const prediction = model.predict(normTestInput);
  const probs = prediction.arraySync()[0];
  const classes = ['≤ 2', '2–10', '> 10'];

  console.log("\nPrediction:");
  probs.forEach((p, i) => {
    console.log(`  ${classes[i]}: ${(p * 100).toFixed(2)}%`);
  });

  const predictedClass = probs.indexOf(Math.max(...probs));
  console.log(`\nPredicted Class: ${classes[predictedClass]}`);
}

let numbers = [];
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

async function askForNumber(count = 1) {
  rl.question(`Enter number ${count}: `, async (input) => {
    const num = parseFloat(input);

    if (isNaN(num)) {
      console.log('Please enter a valid number.');
      askForNumber(count);
    } else {
      numbers.push(num);
      if (numbers.length < 6) {
        askForNumber(count + 1);
      } else {
        await trainAndPredict(numbers);
        numbers = numbers.slice(1)
        askForNumber(6);
        // rl.close();
      }
    }
  });
}

askForNumber();
