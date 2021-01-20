let nn;
let trainSetLen = 200;
let c = trainSetLen;
let disabled = true;
let trainingRate = 1;

let trainSet = [];

function setup() {
    nn = new NN([7, 11, 10]);
    createCanvas(400, 400);
    colorMode(RGB, 1, 1, 1);
    background(0);
    for (let i = 0; i < trainSetLen; i++) {
        trainSet.push(new sevenSeg(floor(random(10))));
    }
}

function toggDis() {
    for (let b of document.getElementsByTagName('BUTTON')) {
        disabled = b.disabled = !b.disabled;
    }
}

function test() {
    loop();
}

function trainNN(n) {
    c = 0;
    trainingRate = n;
    toggDis();
    loop();
}

function draw() {
    background(0);
    if (c < trainSet.length) {
        for (let i = 0; i < trainingRate; i++) {
            let set = random(trainSet);
            nn.train(set.arr, set.corr);
        }
        nn.display();
        push();
            let progress = c / (trainSetLen);
            translate(0, height);
            rotate(-PI / 2);
            noStroke();
            fill(255);
            textSize(40);
            textAlign(CENTER);
            text("Training NN", width/2, height * 0.12);
            textSize(20);
            textAlign(RIGHT);
            fill(0, 255, 255);
            rect(width*0.05 , height * 0.9, width * 0.8 * progress, height * 0.05);
            text(floor(progress * 100) + '%', width - 5, height * 0.948);
            // noStroke();
            // strokeWeight(5);
            // stroke(0,255,255);
            // fill(0,255,255);
            // arc(width/2,height/2,width/3,height/3,-PI/2,3*PI/2 - 2*PI*c/(trainSet.length));
            // noStroke();
            // alph = (sin(2* PI * c/(trainSet.length)))**2;
            // fill(0,255,255,255*alph);
            // textAlign(CENTER);
            // textSize(75);
            // text("Training...",width/2,height*0.6);
        pop();
        c++;
    } else {
        if (disabled) { toggDis(); }
        let tester = random(trainSet);
        nn.test(tester.arr);
        nn.print();
        nn.display();
        nn.showOutput();
        tester.display(10, height - 60, 25, 50);
        noLoop();
    }
}