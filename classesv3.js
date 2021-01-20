class Matrix {
    constructor(m, n, elt = () => 0) {
        this.matrix = [];
        this.dims = [m, n];
        for (let i = 0; i < this.dims[0]; i++) {
            this.matrix.push([]);
            for (let j = 0; j < this.dims[1]; j++) {
                this.matrix[i].push(elt());
            }
        }
    }
    static fromArr(arr) {
        let M = new Matrix(arr.length, arr[0].length);
        for (let i = 0; i < M.dims[0]; i++) {
            if (arr[i].length != M.dims[1]) {
                return;
            }
            for (let j = 0; j < M.dims[1]; j++) {
                M.set(i, j, arr[i][j]);
            }
        }
        return M;
    }
    flatten() {
        let arr = [];
        for (let row of this.matrix) {
            arr = arr.concat(row);
        }
        return arr;
    }
    apply(f) {
        for (let i = 0; i < this.dims[0]; i++) {
            for (let j = 0; j < this.dims[1]; j++) {
                this.set(i, j, f(this.get(i, j)));
            }
        }
    }
    static apply(M, f) {
        let M_ = M.getCopy();
        M_.apply(f);
        return M_;
    }
    applyWith(M, f) {
        if (this.dims.join(',') != M.dims.join(',')) {
            return;
        }
        for (let i = 0; i < this.dims[0]; i++) {
            for (let j = 0; j < this.dims[1]; j++) {
                this.set(i, j, f(this.get(i, j), M.get(i, j)));
            }
        }
    }
    static applyWith(M1, M2, f) {
        let M = M1.getCopy();
        M.applyWith(M2, f);
        return M;
    }
    show(d = 6) {
        let s = ''
        for (let row of this.matrix) {
            for (let elt of row) {
                if (elt != undefined) {
                    s += elt.toFixed(d);
                    s += '\t';
                } else {
                    return;
                }
            }
            s += '\n'
        }
        console.log(s);
        return this.matrix;
    }
    get(i, j) {
        return this.matrix[i][j];
    }
    getRow(i) {
        return Matrix.fromArr([this.matrix[i]]);
    }
    getCol(j) {
        let M = new Matrix(this.dims[0], 1);
        for (let i = 0; i < M.dims[0]; i++) {
            M.set(i, 0, this.get(i, j));
        }
        return M;
    }
    set(i, j, val) {
        this.matrix[i][j] = val;
    }
    getCopy() {
        let M = new Matrix(this.dims[0], this.dims[1]);
        for (let i = 0; i < M.dims[0]; i++) {
            for (let j = 0; j < M.dims[1]; j++) {
                M.set(i, j, this.get(i, j));
            }
        }
        return M;
    }
    copy(M_) {
        this.dims = M_.dims.slice();
        this.matrix = (new Matrix(M_.dims[0], M_.dims[1])).matrix;
        for (let i = 0; i < this.dims[0]; i++) {
            for (let j = 0; j < this.dims[1]; j++) {
                this.set(i, j, M_.get(i, j));
            }
        }
    }
    t() {
        let mT = new Matrix(this.dims[1], this.dims[0]);
        for (let i = 0; i < this.dims[0]; i++) {
            for (let j = 0; j < this.dims[1]; j++) {
                mT.set(j, i, this.get(i, j));
            }
        }
        return mT;
    }
    static hadamard(M1, M2) {
        if (M1.dims.join(',') != M2.dims.join(',')) {
            return;
        }
        let s = 0;
        for (let i = 0; i < M1.dims[0]; i++) {
            for (let j = 0; j < M1.dims[1]; j++) {
                s += M1.get(i, j) * M2.get(i, j);
            }
        }
        return s;
    }
    static cross(M1, M2) {
        if (M1.dims[1] != M2.dims[0]) {
            return;
        }
        let M = new Matrix(M1.dims[0], M2.dims[1]);
        for (let i = 0; i < M.dims[0]; i++) {
            for (let j = 0; j < M.dims[1]; j++) {
                let e = Matrix.hadamard(M1.getRow(i).t(), M2.getCol(j));
                M.set(i, j, e);
            }
        }
        return M;
    }
}

class Layer {
    constructor(curr, prev) {
        this.val = new Matrix(1, curr);
        if (prev) {
            this.w = new Matrix(prev, curr, random);
            this.b = new Matrix(1, curr, random);
        }
    }
}

class NN {
    constructor(nos, f = (x) => 1 / (1 + Math.exp(-x)), df = (x) => x * (1 - x)) {
        this.layers = [];
        for (let i = 0; i < nos.length; i++) {
            this.layers.push(new Layer(nos[i], nos[i - 1]));
        }
        this.correct = this.layers[nos.length - 1].val.getCopy();
        this.f = f;
        this.df = df;
    }
    test(inpArr) {
        this.layers[0].val.copy(Matrix.fromArr([inpArr]));
        this.feedForw();
        return this.layers[this.layers.length - 1].val.show();
    }
    feedForw() {
        for (let i = 1; i < this.layers.length; i++) {
            let l = this.layers[i];
            let d = Matrix.cross(this.layers[i - 1].val, l.w);
            d.applyWith(l.b, (a, b) => a + b);
            d.apply(this.f);
            l.val.copy(d);
        }
    }
    backProp() {
        //2(y-y`)
        let c = Matrix.applyWith(this.correct, this.layers[this.layers.length - 1].val, (a, b) => 2 * (a - b));
        for (let i = this.layers.length - 1; i > 0; i--) {
            let l = this.layers[i];
            c.applyWith(l.val, (a, b) => a * this.df(b));
            l.b.applyWith(c, (a, b) => a + b);
            let d = Matrix.cross(this.layers[i - 1].val.t(), c);
            l.w.applyWith(d, (a, b) => a + b);
            c.copy(Matrix.cross(c, l.w.t()));
        }
    }
    train(inpArr, expArr) {
        this.layers[0].val.copy(Matrix.fromArr([inpArr]));
        this.correct.copy(Matrix.fromArr([expArr]));
        this.feedForw();
        this.backProp();
    }
    print() {
        console.log('***');
        for (let i = 0; i < this.layers.length; i++) {
            let l = this.layers[i];
            console.log('\t\t-' + i + '-');
            for (let p of Object.getOwnPropertyNames(l)) {
                console.log(p);
                l[p].show(3);
            }
        }
        console.log('\t\t-C-')
        this.correct.show(1);
        console.log('***');
    }
    display() {
        let ix = width / (this.layers.length + 1);
        for (let i = 0; i < this.layers.length; i++) {
            let l = this.layers[i];
            let l_ = this.layers[i - 1];
            let iy = height / (l.val.dims[1] + 1);
            for (let j = 0; j < l.val.dims[1]; j++) {
                let val = l.val.get(0, j);
                stroke(1 - val);
                if (l_) {
                    let iy_ = height / (l_.val.dims[1] + 1);
                    for (let k = 0; k < l_.val.dims[1]; k++) {
                        let c = lerpColor(color(1, 0, 0), color(0, 1, 0), this.f(l.w.get(k, j)));
                        strokeWeight(1);
                        stroke(c);
                        line(ix * i, iy_ * (k + 1), ix * (i + 1), iy * (j + 1));
                    }
                    let c = lerpColor(color(1, 0, 0), color(0, 1, 0), this.f(l.b.get(0, j)));
                    stroke(c);
                }
                strokeWeight(2);
                fill(val);
                ellipse(ix * (i + 1), iy * (j + 1), iy / 2, iy / 2);
            }
        }
    }
    showOutput() {
        let output = this.layers[this.layers.length - 1].val.flatten();
        let ans = output.indexOf(max(output));
        ans = ans > 0 ? ans : 0;
        push();
            noStroke();
            fill(255, 255, 0);
            textAlign(CENTER);
            textSize(30);
            text(ans, width * 0.9, (ans + 1.3) * height / (output.length + 1));
        pop();

    }
    cost() {
        let c = Matrix.applyWith(this.correct, this.layers[this.layers.length - 1].val, (a, b) => a - b);
        return Matrix.hadamard(c, c) / this.correct.dims[1];
    }
}

class sevenSeg {
    //   0
    // 1   2
    //   3
    // 4   5
    //   6
    constructor(n) {
        this.arr = sevenSeg.numArr(n);
        this.corr = [];
        for (let i = 0; i < 10; i++) {
            this.corr.push(i == n ? 1 : 0);
        }
    }
    static numArr(n) {
        let nos = {
            0: [1, 1, 1, 0, 1, 1, 1],
            1: [0, 0, 1, 0, 0, 1, 0],
            2: [1, 0, 1, 1, 1, 0, 1],
            3: [1, 0, 1, 1, 0, 1, 1],
            4: [0, 1, 1, 1, 0, 1, 0],
            5: [1, 1, 0, 1, 0, 1, 1],
            6: [1, 1, 0, 1, 1, 1, 1],
            7: [1, 0, 1, 0, 0, 1, 0],
            8: [1, 1, 1, 1, 1, 1, 1],
            9: [1, 1, 1, 1, 0, 1, 1]
        }
        return nos[n];
    }
    display(x, y, w, h) {
        strokeWeight(h / 10);
        stroke(1);
        if (this.arr[0]) {
            line(x, y, x + w, y);
        }
        if (this.arr[1]) {
            line(x, y, x, y + h / 2);
        }
        if (this.arr[2]) {
            line(x + w, y, x + w, y + h / 2);
        }
        if (this.arr[3]) {
            line(x, y + h / 2, x + w, y + h / 2);
        }
        if (this.arr[4]) {
            line(x, y + h / 2, x, y + h);
        }
        if (this.arr[5]) {
            line(x + w, y + h / 2, x + w, y + h);
        }
        if (this.arr[6]) {
            line(x, y + h, x + w, y + h);
        }
    }
}