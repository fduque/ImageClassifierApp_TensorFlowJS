//Setting up the KNN classifier model
const classifier = knnClassifier.create();
const webcamElement = document.getElementById('webcam');
let net;


//Setting up the webcam to capture images
async function setupWebcam() {
    return new Promise((resolve, reject) => {
        const navigatorAny = navigator;
        navigator.getUserMedia = navigator.getUserMedia ||
            navigatorAny.webkitGetUserMedia || navigatorAny.mozGetUserMedia ||
            navigatorAny.msGetUserMedia;
        if (navigator.getUserMedia) {
            navigator.getUserMedia({ video: true },
                stream => {
                    webcamElement.srcObject = stream;
                    webcamElement.addEventListener('loadeddata', () => resolve(), false);
                },
                error => reject());
        } else {
            reject();
        }
    });
}

async function app() {
    console.log('Loading mobilenet..');

    // Load the model.
    net = await mobilenet.load();
    console.log('Sucessfully loaded model');

    await setupWebcam();

    // Reads an image from the webcam and associates it with a specific class
    // index.
    const addExample = classId => {
        // Get the intermediate activation of MobileNet 'conv_preds' and pass that
        // to the KNN classifier.
        const activation = net.infer(webcamElement, 'conv_preds');

        // Pass the intermediate activation to the classifier.
        classifier.addExample(activation, classId);


        const classes = ['A', 'B', 'C'];
        window.alert("Image class "+classes[classId]+" captured!");
    };

    // When clicking a button, add an example for that class.
    document.getElementById('class-a').addEventListener('click', () => addExample(0));
    document.getElementById('class-b').addEventListener('click', () => addExample(1));
    document.getElementById('class-c').addEventListener('click', () => addExample(2));

    while (true) {
        //After the first img captured (>0), start to predict
        if (classifier.getNumClasses() > 0) {
            // Get the activation from mobilenet from the webcam.
            const activation = net.infer(webcamElement, 'conv_preds');
            // Get the most likely class and confidences from the classifier module.
            const result = await classifier.predictClass(activation);

            const classes = ['A', 'B', 'C'];
            document.getElementById('console').innerText = `
                prediction: ${classes[result.classIndex]}
                probability: ${result.confidences[result.classIndex]}
                `;
        }

        await tf.nextFrame();
    }
}

app();

