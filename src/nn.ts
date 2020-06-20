import V = require("vectorious");

export type AF = "sigmoid" | "relu" | "tanh" | "leakyRelu";
export type Layer = {
	w: V;
	b: V;
	cache?: {
		Z: V;
		A_prev: V;
	};
};
const activeFcts: [AF, AF] = ["sigmoid", "sigmoid"];
export default class NeuralNetwork {
	layers: Layer[];
	activation_functions: AF[];
	learning_rate: number;
	examples: number;
	constructor(
		structure: number[],
		learning_rate = 0.01,
		examples = 1,
		AF: AF[] = ["sigmoid"]
	) {
		this.layers = NeuralNetwork.initializeParams(structure);
		this.activation_functions = AF;
		this.learning_rate = learning_rate;
		this.examples = examples;
	}

	guess(x: number[] | number[][]) {
		let A = V.array(x).reshape(
			x.length,
			typeof x[0] == "object" ? x[0].length : 1
		);
		this.layers.forEach((layer, i) => {
			// linear Z = W * X + b
			const Z = this.linearForward(layer.w, layer.b, A);

			layer.cache = { Z, A_prev: A };

			// activation
			const isLast = i == this.layers.length - 1;
			A = this.linearActivationForward(
				Z,
				isLast ? activeFcts[1] : activeFcts[0]
			);
		});

		return A;
	}

	train(X, y, printCost = false) {
		return new Promise<number>((res) => {
			const Y = V.array(y).reshape(
				this.layers[this.layers.length - 1].w.shape[0],
				this.examples
			);

			//generate the output of last layer (Output)
			const AL = this.guess(X);
			const cost = this.computeCost(AL, Y);
			// compute the first Derivitive of A (Derivitive of lost function)

			let dA_prev = this.lastLayerDerivitive(AL, Y);

			Array.from(this.layers)
				.reverse()
				.forEach((layer, i) => {
					const isLast = i == 0;
					//compute dZ
					const Gprim = this.linearActivationBackword(
						layer.cache.Z,
						isLast ? activeFcts[1] : activeFcts[0]
					);
					const dZ = V.product(dA_prev, Gprim);

					const gards = this.linearBackward(
						layer.cache.A_prev,
						layer.w,
						dZ
					);

					const { dW, db } = gards;
					dA_prev = gards.dA_prev;

					layer.w.subtract(V.scale(dW, this.learning_rate));
					layer.b.subtract(V.scale(db, this.learning_rate));
				});
			if (printCost) console.log(`cost: ${cost}`);
			res(cost);
		});
	}

	linearForward(W: V, b: V, A: V) {
		const termA = V.multiply(W, A);
		const boradCast_b = V.multiply(b, V.ones(1, termA.shape[1]));

		const Z = termA.add(boradCast_b);
		if (Z.shape[0] != W.shape[0] || Z.shape[1] != A.shape[1])
			throw Error("Z shape error in Linear Forwards");

		return Z;
	}

	linearActivationForward(Z: V, activation_functions: AF) {
		if (activation_functions == "sigmoid")
			return V.map(Z, NeuralNetwork.sigmoid);
		else if (activation_functions == "relu")
			return V.map(Z, NeuralNetwork.relu);
		else if (activation_functions == "tanh")
			return V.map(Z, NeuralNetwork.tanh);
		else if (activation_functions == "leakyRelu")
			return V.map(Z, NeuralNetwork.leakyRelu);
	}
	linearActivationBackword(Z: V, activation_functions: AF) {
		if (activation_functions == "sigmoid")
			return V.map(Z, NeuralNetwork.sigmoidPrim);
		else if (activation_functions == "relu")
			return V.map(Z, NeuralNetwork.reluPrime);
		else if (activation_functions == "tanh")
			return V.map(Z, NeuralNetwork.tanhPrime);
		else if (activation_functions == "leakyRelu")
			return V.map(Z, NeuralNetwork.leakyReluPrime);
	}

	linearBackward(A_prev: V, W: V, dZ: V) {
		const m = A_prev.shape[1]; // examples

		const dW = V.multiply(dZ, A_prev.T).scale(1 / m);
		const db = V.multiply(dZ, V.ones(dZ.shape[1], 1)).scale(1 / m);
		const dA_prev = V.multiply(W.T, dZ);
		return { dW, db, dA_prev };
	}

	lastLayerDerivitive(AL: V, Y: V) {
		// dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL)) # derivative of cost with respect to AL
		const termA = V.product(Y, V.pow(AL, -1)); // np.divide(Y, AL)

		const Ones = V.ones(...Y.shape); // 1
		const OneMinusY = V.subtract(Ones, Y); // 1 - Y
		const OneMinusAL = V.subtract(Ones, AL); // 1 - AL

		const termB = V.product(OneMinusY, V.pow(OneMinusAL, -1)); // np.divide(1 - Y, 1 - AL)

		return termA.subtract(termB).scale(-1);
	}

	computeCost(AL: V, Y: V) {
		// cost = -(1/m) * np.sum( Y * np.log(AL) + (1 - Y) * np.log(1 - AL) ,axis=1)
		const Ones = V.ones(...Y.shape);
		const OneMinusY = V.subtract(Ones, Y);
		const OneMinusAL = V.subtract(Ones, AL);
		const m = Y.shape[1]; // number of examples

		const cost = V.product(Y, V.log(AL)).add(
			V.product(OneMinusY, V.log(OneMinusAL))
		);
		// calculat sum
		const sumHorizontal = cost.multiply(V.ones(m, 1)).scale(1 / m);
		const sumVertical = sumHorizontal.sum() / AL.shape[0];
		return Math.abs(sumVertical);
	}

	static sigmoid(x: number) {
		return 1 / (1 + Math.exp(-x));
	}
	static sigmoidPrim(x: number) {
		return NeuralNetwork.sigmoid(x) * (1 - NeuralNetwork.sigmoid(x));
	}

	static relu(x: number) {
		return Math.max(0, x);
	}
	static reluPrime(x: number) {
		return x >= 0 ? 1 : 0;
	}

	static leakyRelu(x: number) {
		return Math.max(0.01 * x, x);
	}
	static leakyReluPrime(x: number) {
		return x >= 0 ? 1 : 0.01;
	}

	static tanh(x: number) {
		return Math.tanh(x);
	}
	static tanhPrime(x: number) {
		return Math.atanh(x);
	}

	static initializeParams(structure, lower_weights = 0.01) {
		const layers: Layer[] = [];
		for (var i = 1; i < structure.length; i++) {
			layers.push({
				w: V.random(structure[i], structure[i - 1])
					.scale(2)
					.subtract(V.ones(structure[i], structure[i - 1]))
					.scale(lower_weights),
				b: V.zeros(structure[i], 1),
			});
		}
		return layers;
	}
	static IndexOfmaxActive(activations: number[]) {
		let Max = [0, 0];
		for (let i = 0; i < activations.length; i++) {
			if (activations[i] > Max[0]) Max = [activations[i], i];
		}
		return Max[1];
	}
}
