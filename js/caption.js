
let img = document.querySelector("#image");
let isModelLoaded = false;

let model;

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

async function start() {
    model = await tf.loadModel('model/model.json');
    model.predict(tf.zeros([1,224,224,3])); 
}