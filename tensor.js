const tf = require('@tensorflow/tfjs');
const readline = require('readline-sync');
// --- Hyperparameters ---
const SEQ_LEN = 4;
const EPOCHS = 10;
const LEARNING_RATE = 0.01;

// --- Data Storage ---
let trainingData = [
    [2.37, 1.47, 2.88, 1.33, 6.66,  3.25,  1.24, 8.27, 4.46, 7.93],
    [1.65, 2.09, 2.00, 9.24, 33.53, 10.88, 1.33, 2.10, 1.14, 1.20],
    [3.8,  3.14, 1.44, 5.28, 1.85,  1.10,  20.26,1.31, 1.21, 3.33],
    [1.71, 1.62, 1.10, 1.69, 2.31,  1.02,  1.58, 10.2, 1.00, 1.12], 
    [15.52,2.05, 6.62, 5.61, 3.18,  1.61,  7.89, 53.14,1.7,  1.64], 
    [1.28, 5.46, 2.78, 1.87, 1.37,  2.64,  1.57, 1.57, 1.65, 4.19], 
    [1.9,  1.05, 3.04, 1.0,  1.0,   2.16,  4.04, 2.24, 11.26,1.12], 
    [3.69, 3.55, 1.79, 4.13, 17.99, 5.54,  1.46, 2.7,  14.57, 50.21],
];

// --- Create the Model ---
function createModel() {
  const model = tf.sequential();
  model.add(tf.layers.lstm({
    units: 50,
    inputShape: [SEQ_LEN, 1],
    returnSequences: false
  }));
  model.add(tf.layers.dense({ units: 1 }));
  model.compile({
    optimizer: tf.train.adam(LEARNING_RATE),
    loss: 'meanSquaredError'
  });
  return model;
}

// --- Preprocess Data ---
function createTensors(data) {
  const xs = [];
  const ys = [];
  data.forEach(seq => {
    const input = seq.slice(0, SEQ_LEN);
    const output = seq[SEQ_LEN];
    xs.push(input.map(x => [x])); // shape: [SEQ_LEN, 1]
    ys.push([output]);            // shape: [1]
  });
  return {
    xs: tf.tensor3d(xs), // shape: [batch, SEQ_LEN, 1]
    ys: tf.tensor2d(ys)  // shape: [batch, 1]
  };
}

// --- Train Model ---
async function trainModel(model, data) {
  const { xs, ys } = createTensors(data);
  await model.fit(xs, ys, {
    epochs: EPOCHS,
    verbose: 0
  });
  xs.dispose();
  ys.dispose();
}

// --- Predict Next Number ---
function predictNext(model, inputSeq) {
  const input = tf.tensor3d([inputSeq.map(x => [x])]);
  const prediction = model.predict(input);
  const result = prediction.dataSync()[0];
  input.dispose();
  prediction.dispose();
  return result;
}

// --- Main Loop ---
async function runInteractive() {
  const model = createModel();
  await trainModel(model, trainingData);
  console.log("✅ Model trained on initial data.\n");

  while (true) {
    // Get 4 numbers from user
    const userInput = [];
    for (let i = 0; i < SEQ_LEN; i++) {
      const num = readline.questionFloat(`Enter number ${i + 1} of ${SEQ_LEN}: `);
      userInput.push(num);
    }

    // Predict
    const prediction = predictNext(model, userInput);
    console.log(`🤖 Predicted next number: ${prediction.toFixed(2)}`);

    // Get actual next number from user
    const actual = readline.questionFloat(`✅ Enter actual next number: `);

    // Retrain with this new sequence
    const newSequence = userInput.concat(actual);
    trainingData.push(newSequence);
    await trainModel(model, trainingData);
    console.log("🔁 Model updated with your feedback.\n");
  }
}

runInteractive();