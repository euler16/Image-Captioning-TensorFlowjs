
let img = document.querySelector("#image");
let button = document.querySelector("#btn");
let text = document.querySelector("#txt");
let capField = document.querySelector("#caption");
let isModelLoaded = false;

let model;
let mobileNet;

const maxLen = 3; // 40

function preprocess(imgElement) {
    return tf.tidy(() => {
        let tensor = tf.fromPixels(imgElement).toFloat();
        const resized = tf.image.resizeBilinear(tensor,[224,224]);
        const offset = 125.5;
        const normalized = resized.div(offset).sub(tf.scalar(1.0));
        const batched = normalized.expandDims(0);
        return batched;
    });
}

function caption(img) {
    // should use promise and async-await to make it non blocking
    // max_len change karna
    console.log("Inside caption()");
    return tf.tidy(()=> {
        let startWord = ['<start>'];
        while (true) {
            let parCaps = [];
            for (let j = 0; j < startWord.length; ++j) {
                parCaps.append(word2idx[start_word[j]]);
            }
            parCaps = tf.tensor1d(parCaps).pad([[0, maxLen - startWord.length]]);
            let e = mobileNet.predict(img);
            let preds = model.predict([e,par_caps]);
            let wordPred = idx2word[preds.argMax()];
            startWord.append(wordPred);

            if(wordPred=='<end>'||startWord.length>maxLen)
                break;
        }
        // removing first and last tokern <start> and <end>
        startWord.shift();
        startWord.pop();

        return startWord.join(' ');
    }); 
}

async function loadMobileNet() {
    const mobilenet = await tf.loadModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');
    const layer = mobilenet.getLayer('conv_preds');
    console.log("mobileNet loaded");
    return tf.model({
        'inputs': mobilenet.inputs,
        'outputs': layer.output
    });
    
}

async function start() {
    mobileNet = loadMobileNet();
    model = await tf.loadModel('model/model.json');
    console.log("Inside start()");
    mobileNet.predict(tf.zeros([1,224,224,3]));
    modelLoaded();
}

function modelLoaded() {
    console.log("Inside modelLoaded()");
    isModelLoaded = true;
    text.innerHTML = "Models Loaded!";
}

button.addEventListener("click",function() {
    console.log("button pressed");
    if(!isModelLoaded) {
        console.log('Models not loaded yet');
        return;
    }
    let picture = preprocess(img);
    let cap = caption(picture);
    capField.innerHTML = cap;
});

start();
