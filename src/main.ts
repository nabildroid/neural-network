import NN from "./nn";

const examples = 300;
const Brain = new NN([1, 20, 20,20,20, 1], 0.001, examples);

const training: Promise<number>[] = [];
for (let i = 0; i < 5000; i++) {
	const X: number[] = [];
	const Y: number[] = [];
	while (Y.length < examples) {
		const x = Math.random();
		const y = Math.sin(x);
		X.push(x);
		Y.push(Math.abs(y));
	}
	training.push(Brain.train([X], Y, i % 100 == 0));
}

Promise.all(training).then(() => {
	for (let i = 0; i < 10; i++) {
		const x = Math.random();
		const guess = Brain.guess([x]).toArray()[0][0] * 2;

		console.log(`sin(${x.toFixed(4)})= ${guess.toFixed(4)}`);
	}
});
