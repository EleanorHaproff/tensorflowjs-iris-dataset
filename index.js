(async () => {

    tf.util.shuffle(irisData)

    const data = tf.tensor2d(irisData.map(({ sepalLength, sepalWidth, petalLength, petalWidth }) => [
        sepalLength, sepalWidth, petalLength, petalWidth
    ]));


    const normalisedData = getNormalisedTensors(data);
    const labels = tf.tensor2d(irisData.map(({ species }) => [
        species === 'setosa' ? 1 : 0,
        species === 'virginica' ? 1 : 0,
        species === 'versicolor' ? 1 : 0,
    ]));

    const model = tf.sequential();

    model.add(tf.layers.dense({
        inputShape: [4],
        activation: 'relu',
        units: 15,
    }));

    model.add(tf.layers.dense({
        activation: 'relu',
        units: 15,
    }));

    model.add(tf.layers.dense({
        activation: 'softmax',
        units: 3
    }));

    model.compile({
        optimizer: 'sgd',
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });

    const results = await model.fit(
        normalisedData,
        labels,
        {
            epochs: 100,
            validationSplit: 0.2
        }
    );

    console.log(results.history);


    const pred = model.predict(tf.tensor2d([[5.6, 2.7, 4.2, 1.3]])
    pred.print()
    //   {"sepalLength": 5.6, "sepalWidth": 2.7, "petalLength": 4.2, "petalWidth": 1.3, "species": "versicolor"},

})();


function getNormalisedTensors(tensor) {
    const max = tensor.max(0);
    const min = tensor.min(0);
    return tensor.sub(min).div(max.sub(min));
}
