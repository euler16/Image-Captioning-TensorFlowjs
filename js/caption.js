
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
    return tf.model({
        'inputs': mobilenet.inputs,
        'outputs': layer.output
    });
}

async function start() {
    mobileNet = loadMobileNet();
    model = await tf.loadModel('model/model.json');
    mobileNet.predict(tf.zeros([1,224,224,3]));
    
}