const brain = require('brain.js');
const readline = require('readline');

// Create LSTM TimeStep network
const net = new brain.recurrent.LSTMTimeStep();

// Initial training data
const trainingData = [
  [1, 2, 3, 4, 5],
  [10, 20, 30, 40, 50],
  [100, 200, 300, 400, 500],
];

net.train(trainingData, {
  learningRate: 0.005,
  errorThresh: 0.02,
  iterations: 2000
});

// Setup readline interface for user input
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

// Loop function to ask user for actual values
function askForNextInput() {
  const inputSequence = [];

  // Ask user to input a sequence of numbers
  function askSequence(index) {
    if (index < 4) {
      rl.question(`Enter number ${index + 1} of 4 in sequence: `, (answer) => {
        inputSequence.push(Number(answer));
        askSequence(index + 1);
      });
    } else {
      const prediction = net.run(inputSequence);
      console.log(`🤖 Predicted next number: ${prediction.toFixed(2)}`);

      rl.question(`✅ Enter actual next number (ground truth): `, (actualAnswer) => {
        const actual = Number(actualAnswer);
        const error = Math.abs(actual - prediction);
        console.log(`📉 Error: ${error.toFixed(2)}\n🔁 Retraining...`);

        // Retrain with user-confirmed data
        net.train([inputSequence.concat(actual)], {
          iterations: 100,
          errorThresh: 0.01,
          log: false
        });

        console.log('✅ Model updated based on your feedback.\n');
        askForNextInput(); // Repeat the loop
      });
    }
  }

  askSequence(0);
}

// Start the interactive loop
askForNextInput();