const tf = require('@tensorflow/tfjs');
const rawNumbers = require('./data.js');
// Raw number sequence
// const rawNumbers = [1.2, 1.9, 3.1, 5.6, 1.4, 2.0, 7.8, 1.5, 3.2, 1.1, 4.3, 1.8];

// Convert to binary: 0 for [1–2], 1 for >2
const binarySequence = rawNumbers.map(n => (n > 2 ? 1 : 0));

// Parameters
const sequenceLength = 3;
const xs = [];
const ys = [];

// Prepare training data
for (let i = 0; i < binarySequence.length - sequenceLength; i++) {
  const input = binarySequence.slice(i, i + sequenceLength);
  const output = binarySequence[i + sequenceLength];
  xs.push(input);
  ys.push(output);
}

// Convert to tensors
const xsTensor = tf.tensor2d(xs); // shape: [samples, sequenceLength]
const ysTensor = tf.tensor2d(ys, [ys.length, 1]); // shape: [samples, 1]

// Build model
const model = tf.sequential();
model.add(tf.layers.dense({ units: 10, activation: 'relu', inputShape: [sequenceLength] }));
model.add(tf.layers.dense({ units: 5, activation: 'relu' }));
model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

// Compile model
model.compile({
  optimizer: tf.train.adam(0.01),
  loss: 'binaryCrossentropy',
  metrics: ['accuracy'],
});

// Train the model
(async () => {
  await model.fit(xsTensor, ysTensor, {
    epochs: 100,
    verbose: 0,
  });

  // Predict next range
  const recentSequence = binarySequence.slice(-sequenceLength);
  const inputTensor = tf.tensor2d([recentSequence]);
  const prediction = await model.predict(inputTensor).data();

  const probability = prediction[0];
  console.log(`Prediction (0 = [1–2], 1 = >2):`, probability.toFixed(4));
  console.log(probability > 0.5 ? 'Likely >2' : 'Likely between 1 and 2');
})();