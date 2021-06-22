const fs = require("fs");
const path = require("path");
const cocoSsd = require("@tensorflow-models/coco-ssd");
const tf = require("@tensorflow/tfjs-node");
const Canvas = require("canvas");
const Image = Canvas.Image;

const wait = () => {
  return new Promise((resolve) => {
    setTimeout(() => {
      resolve();
    }, 100);
  })
}

(async () => {
  const iDir = "./VID_20210620_165856";
  const fileNames = (await fs.promises.readdir(iDir)).filter(
    (name) => path.parse(name).ext === ".png"
  );

  // Load the model.
  const model = await cocoSsd.load();

  for (const filename of fileNames) {
    const filepath = path.join(iDir, filename);
    const img = await fs.promises.readFile(filepath);
    const imgTensor = tf.node.decodeImage(new Uint8Array(img), 3);

    // Classify the image.
    const predictions = await model.detect(imgTensor);

    imgTensor.dispose();

    console.log(`${filename} Predictions: ${predictions.map(p => p.class).join(', ')}`);

    const newImg = new Image();
    newImg.src = img;
    const canvas = new Canvas.Canvas(newImg.width, newImg.height);
    const ctx = canvas.getContext("2d");
    ctx.drawImage(newImg, 0, 0, newImg.width, newImg.height);

    ctx.strokeStyle = "blue";
    ctx.lineWidth = "10";
    ctx.fillStyle = "red";
    ctx.font = '30px serif';
    const textMaxWidth = 100;

    for (const prediction of predictions) {
      ctx.strokeRect(
        prediction.bbox[0],
        prediction.bbox[1],
        prediction.bbox[2],
        prediction.bbox[3]
      );

      ctx.fillText(`${prediction.class}: ${Math.round(prediction.score * 100)}%`, prediction.bbox[0] + (prediction.bbox[2] / 2) - (textMaxWidth / 2), prediction.bbox[1] - 10, textMaxWidth);
    }

    const base64String = canvas.toDataURL().split(",")[1];
    const buffer = Buffer.from(base64String, "base64");
    await fs.promises.writeFile(
      path.join("VID_20210620_165856_detected_objects", filename),
      buffer
    );

    delete ctx
    delete canvas
    newImg.src = null;

    if (global.gc) {
      global.gc();
    }

    await wait();
  }

  model.dispose();
})();
