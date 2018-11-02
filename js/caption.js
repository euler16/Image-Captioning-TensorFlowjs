
let img = document.querySelector("#image");
let isModelLoaded = false;

let model;
let mobileNet;

const maxLen = 3; // 40

function preprocess(img) {
    return tf.tidy(() => {
        let tensor = tf.fromPixels(imgData).toFloat();
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
    
    let start_word = ['<start>'];
    while(true) {
        let par_caps = [];
        for(let j=0;j<start_word.length;++j) {
            par_caps.append(word2idx[start_word[j]]);
        }
        par_caps = tf.tensor1d(par_caps).pad([[0,maxLen-start_word.length]]);
        let e = mobileNet.predict()
    }    
}

async function loadMobileNet() {
    const mobilenet = await tf.loadModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');
    const layer = mobilenet.getLayer('conv_preds');
    return tf.model({
        'inputs': mobilenet.inputs,
        'outputs': layer.output
    });
}

async function start() {
    mobileNet = loadMobileNet();
    model = await tf.loadModel('model/model.json');
    model.predict(tf.zeros([1,224,224,3])); 
}