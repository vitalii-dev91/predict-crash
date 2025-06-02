const tf = require('@tensorflow/tfjs');
const readline = require('readline-sync');
const trainingData = require('./data.js');

// --- Hyperparameters ---
const EPOCHS = 10;
const LEARNING_RATE = 0.01;


// Normalize input values using log transformation for values above 1
function normalizeData(value) {
  if (value < 1) {
    // Clip values below 1 to be treated as 1
    return 0;
  } else if (value <= 2) {
    // For values between 1 and 2, use linear scaling
    return (value - 1) / 1;
  } else {
    // For values greater than 2, use logarithmic scaling to compress the range
    return Math.log(value - 1) / Math.log(100); // Logarithm with base 100 for better compression
  }
}

// Inverse transformation for predicting actual values
function inverseNormalizeData(value) {
  if (value === 0) {
    return 1;
  } else if (value < 1) {
    return 1 + Math.exp(value * Math.log(100));
  } else {
    return 1 + value;
  }
}

// --- Create the Model ---
function createModel() {
  const model = tf.sequential();
  model.add(tf.layers.dense({
    units: 50,
    inputShape: [1], // Only one input feature (independent number)
    activation: 'relu'
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
  for (let i = 0; i < data.length - 1; i++) {
    xs.push([normalizeData(data[i])]); // shape: [1] - independent number
    ys.push([normalizeData(data[i + 1])]); // shape: [1] - next number
  }
  return {
    xs: tf.tensor2d(xs), // shape: [batch, 1]
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
function predictNext(model, inputNum) {
  const input = tf.tensor2d([[normalizeData(inputNum)]]);
  const prediction = model.predict(input);
  const result = prediction.dataSync()[0];
  input.dispose();
  prediction.dispose();
  return inverseNormalizeData(result);
}

// --- Save Model ---
async function saveModel(model) {
  await model.save('file://./my-trained-model');
  console.log("💾 Model saved.");
}

// --- Load Model ---
async function loadModel() {
  return await tf.loadLayersModel('file://./my-trained-model/model.json');
}

// --- Main Loop ---
async function runInteractive() {
  let model;
  // Try to load the model if it exists, otherwise create a new one
//   try {
//     model = await loadModel();
//     console.log("✅ Model loaded.");
//   } catch (error) {
//     console.log("🚫 No existing model found. Creating a new one.");
    model = createModel();
    await trainModel(model, trainingData);  // Train with initial data
//     console.log("✅ Model trained on initial data.");
//   }

  while (true) {
    // Get 1 number from user
    const userInput = readline.questionFloat("Enter a number between 1 and positive infinity: ");

    // Predict
    const prediction = predictNext(model, userInput);
    console.log(`🤖 Predicted next number: ${prediction.toFixed(2)}`);

    // Get actual next number from user
    const actual = readline.questionFloat("✅ Enter actual next number: ");

    // Retrain with the new data
    trainingData.push(userInput, actual);

    await trainModel(model, trainingData);
    console.log("🔁 Model updated with your feedback.\n");

    // Optionally save after each update
    // await saveModel(model);
  }
}

runInteractive();