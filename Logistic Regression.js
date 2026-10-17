class Matrix {
    #data;
    #shape;

    constructor(data, shape) {
        this.#data = data;
        this.#shape = shape;
    }

    get shape() { return this.#shape; }
    
    findElement(row, col) { return this.#data[this.#shape[1] * row + col]; }
    transpose() {
        const data = [];
        for (let i = 0; i < this.#shape[1]; i++) {
            for (let j = 0; j < this.#shape[0]; j++) {
                data.push(this.findElement(j, i));
            }
        }

        return new Matrix(data, [this.#shape[1], this.#shape[0]]);
    }
    multiply(matrix) {
        let resultMatrix;
        if (this.#shape[1] === matrix.#shape[0]) {
            const data = [];
            for (let i = 0; i < this.#shape[0]; i++) {
                for (let j = 0; j < matrix.#shape[1]; j++) {
                    let value = 0;
                    for (let k = 0; k < this.#shape[1]; k++) {
                        value+=this.findElement(i, k) * matrix.findElement(k, j);
                    }

                    data.push(value);
                }
            }

            resultMatrix = new Matrix(data, [this.#shape[0], matrix.#shape[1]]);
        }

        return resultMatrix;
    }
    multiplyElementWise(matrix) {
        let resultMatrix;
        if (this.#shape[0] === matrix.#shape[0] && this.#shape[1] === matrix.#shape[1]) {
            const data = [];
            for (let i = 0; i < this.#data.length; i++) {
                data.push(this.#data[i] * matrix.#data[i]);
            }

            resultMatrix = new Matrix(data, [this.#shape[0], this.#shape[1]]);
        }

        return resultMatrix;
    }
    add(matrix) {
        let resultMatrix;
        if (this.#shape[0] === matrix.#shape[0] && this.#shape[1] === matrix.#shape[1]) {
            const data = [];
            for (let i = 0; i < this.#data.length; i++) {
                data.push(this.#data[i] + matrix.#data[i]);
            }

            resultMatrix = new Matrix(data, [this.#shape[0], this.#shape[1]]);
        }

        return resultMatrix;
    }
    subtract(matrix) {
        let resultMatrix;
        if (this.#shape[0] === matrix.#shape[0] && this.#shape[1] === matrix.#shape[1]) {
            const data = [];
            for (let i = 0; i < this.#data.length; i++) {
                data.push(this.#data[i] - matrix.#data[i]);
            }

            resultMatrix = new Matrix(data, [this.#shape[0], this.#shape[1]]);
        }

        return resultMatrix;
    }
    multiplyScalar(scalar) {
        const data = [];
        for (let i = 0; i < this.#data.length; i++) {
            data.push(this.#data[i] * scalar);
        }

        return new Matrix(data, [this.#shape[0], this.#shape[1]]);
    }
    addScalar(scalar) {
        const data = [];
        for (let i = 0; i < this.#data.length; i++) {
            data.push(this.#data[i] + scalar);
        }

        return new Matrix(data, [this.#shape[0], this.#shape[1]]);
    }
    subRows(lowerBound, upperBound) {
        const data = [];
        for (let i = lowerBound; i < upperBound; i++) {
            for (let j = 0; j < this.#shape[1]; j++) {
                data.push(this.findElement(i, j));
            }
        }

        return new Matrix(data, [upperBound - lowerBound - 1, this.#shape[1]]);
    }
    sigmoid() {
        const data = [];
        for (let i = 0; i < this.#data.length; i++) {
            data.push(1 / (1 + Math.exp(-this.#data[i])));
        }

        return new Matrix(data, [this.#shape[0], this.#shape[1]]);
    }
    binaryCrossEntropy(prediction) {
        let loss = 0;
        for (let i = 0; i < this.#data.length; i++) {
            loss+=(-this.#data[i]) * Math.log(prediction.#data[i]) - (1 - this.#data[i]) * Math.log(1 - prediction.#data[i]);
        }

        return (1 / this.#data.length) * loss;
    }
    average() {
        let summation = 0;
        for (let i = 0; i < this.#data.length; i++) {
            summation+=this.#data[i];
        }

        return (1 / this.#data.length) * summation;
    }

    static zeros(shape) {
        const data = [];
        for (let i = 0; i < shape[0] * shape[1]; i++) { data.push(0); }

        return new Matrix(data, shape);
    }
}

class GenericLogisticRegression {
    #x;
    #y;
    #b0;
    #b;
    #b0Momentum;
    #bMomentum;

    constructor(x, y) {
        this.#x = x;
        this.#y = y;
        this.#b0 = 0;
        this.#b = Matrix.zeros([this.#x.shape[1], 1]);
    }

    #z(matrix) { return matrix.multiply(this.#b).addScalar(this.#b0); }
    #a(matrix) { return this.#z(matrix).sigmoid(); }
    #gradientDescent(epochs, batches, momentum, learningRate, l2Rate) {
        const points = [];

        for (let i = 0; i < epochs; i++) {
            const numBatches = Math.ceil(this.#x.shape[0] / batches);
            for (let j = 0; j < numBatches; j++) {
                const lowerBound = j * batches;
                const upperBound = Math.min(lowerBound + (j + 1) * batches, this.#x.shape[0]);
                const x = this.#x.subRows(lowerBound, upperBound);
                const y = this.#y.subRows(lowerBound, upperBound);

                const ay = this.#a(x).subtract(y);
                const dldb0 = learningRate * (ay.average() + 2 * l2Rate * this.#b0);
                const dldb = x.transpose().multiply(ay).add(this.#b.multiplyScalar(l2Rate)).multiplyScalar(learningRate);

                if (!this.#b0Momentum && !this.#bMomentum) {
                    this.#b0Momentum = dldb0;
                    this.#bMomentum = dldb;
                } else {
                    this.#b0Momentum = momentum * this.#b0Momentum + dldb0;
                    this.#bMomentum = this.#bMomentum.multiplyScalar(momentum).add(dldb);
                }

                this.#b0-=this.#b0Momentum;
                this.#b = this.#b.subtract(this.#bMomentum);
                
                points.push(new Point(i * numBatches + j, this.loss()));
            }
        }

        return points;
    }

    loss() { return this.#y.binaryCrossEntropy(this.#a(this.#x)); }
    train(epochs = 5, batches = 16, momentum = 0.9, learningRate = 0.001, l2Rate = 0.001) { return this.#gradientDescent(epochs, batches, momentum, learningRate, l2Rate); }
    test(data) { return this.#a(data); }
}
