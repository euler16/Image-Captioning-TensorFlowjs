
let img = document.querySelector("#image");
let button = document.querySelector("#btn");
let text = document.querySelector("#txt");
let capField = document.querySelector("#caption");
let imageLoader = document.querySelector("#imageLoader");

let isModelLoaded = false;

let model;
let mobileNet;

const maxLen = 40; // 40

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

/* function asyncCaption(img) {

    let startWord = ['<start>'];
    let idx, wordPred;
    let flattenLayer = tf.layers.flatten();
    let e = flattenLayer.apply(mobileNet.predict(img));
    function captionLoop(callback) {
        let continueLoop = true;
        
        function whileBlock() {
            if(continueLoop) {
                let parCaps = [];
                for (let j = 0; j < startWord.length; ++j) {
                    //console.log(typeof(parCaps));
                    parCaps.push(word2idx[startWord[j]]);
                }
                parCaps = tf.tensor1d(parCaps)
                    .pad([[0, maxLen - startWord.length]])
                    .expandDims(0);

                
                let preds = model.predict([e, parCaps]);
                preds = preds.reshape([preds.shape[1]]);
                //console.log(preds.shape);

                idx = preds.argMax().dataSync();
                wordPred = idx2word[idx];
                console.log(wordPred);
                startWord.push(wordPred);
                if (wordPred == '<end>' || startWord.length > maxLen)
                    continueLoop = false;    

                whileBlock();
            } else {
                return callback();
            }
        }
        whileBlock();
    }

    function postCaption() {
        startWord.shift();
        startWord.pop();
        cap = startWord.join(' ');
        console.log("caption: ", cap);
        capField.innerHTML = cap;
        return cap;
    }

    return captionLoop(postCaption);
} */

function caption(img) {
    // should use promise and async-await to make it non blocking
    // max_len change karna
    let flattenLayer = tf.layers.flatten();
    console.log("Inside caption()");
    return tf.tidy(()=> {

        let startWord = ['<start>'];
        let idx,wordPred;
        let e = flattenLayer.apply(mobileNet.predict(img));
        
        while (true) {
            let parCaps = [];
            for (let j = 0; j < startWord.length; ++j) {
                //console.log(typeof(parCaps));
                parCaps.push(word2idx[startWord[j]]);
            }
            parCaps = tf.tensor1d(parCaps)
                        .pad([[0, maxLen - startWord.length]])
                        .expandDims(0);

            
            let preds = model.predict([e,parCaps]);
            preds = preds.reshape([preds.shape[1]]);
            //console.log(preds.shape);
            
            idx = preds.argMax().dataSync();
            wordPred = idx2word[idx];
            
            startWord.push(wordPred);            
            if(wordPred=='<end>'||startWord.length>maxLen)
                break;
        }
        // removing first and last tokern <start> and <end>
        startWord.shift();
        startWord.pop();
        
        cap = startWord.join(' ');
        console.log("caption: ", cap);
        capField.innerHTML = cap;
        //return startWord.join(' ');
        //return asyncCaption(img);
    }); 
}


async function start() {
    //mobileNet = loadMobileNet();
    const mobilenet = await tf.loadModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');
    const layer = mobilenet.getLayer('conv_preds');
    //console.log("mobileNet loaded");
    mobileNet =  tf.model({
        'inputs': mobilenet.inputs,
        'outputs': layer.output
    });

    model = await tf.loadModel('model/model.json');
    console.log("Inside start()");
    modelLoaded();
}

function modelLoaded() {
    mobileNet.predict(tf.zeros([1, 224, 224, 3])).dispose();
    //console.log("Inside modelLoaded()");
    isModelLoaded = true;
    text.innerHTML = "Models Loaded!";
}


imageLoader.addEventListener('change', function(e) {
    let reader = new FileReader();
    reader.onload = function() {
        img.src = reader.result;
    }
    reader.readAsDataURL(e.target.files[0]); 
}, false);

button.addEventListener("click",function() {
    //console.log("button pressed");
    if(!isModelLoaded) {
        console.log('Models not loaded yet');
        return;
    }
    capField.innerHTML = "Generating Caption ... Please Wait";
    let picture = preprocess(img);
    //let cap = caption(picture);
    caption(picture);
    //console.log("caption: "+ cap);
    //capField.innerHTML = cap;
});

start();
