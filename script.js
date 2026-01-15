let selectVideo = document.getElementById('selectVideo');
let selectCanvas = document.getElementById('selectCanvas');
let selectCtx = selectCanvas.getContext('2d');
let searchVideo = document.getElementById('searchVideo');
let searchCanvas = document.getElementById('searchCanvas');
let searchCtx = searchCanvas.getContext('2d');
let model;
let mobilenetModel;
let referenceFeatures = null;
let referenceClass = null;
let isSearching = false;

async function loadModels() {
    model = await cocoSsd.load();
    mobilenetModel = await mobilenet.load();
    console.log('Modelos carregados');
}

document.getElementById('startSelect').addEventListener('click', async () => {
    const facingMode = document.getElementById('selectCamera').value;
    const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode } });
    selectVideo.srcObject = stream;
    document.getElementById('captureSelect').style.display = 'inline';
});

document.getElementById('captureSelect').addEventListener('click', () => {
    selectCanvas.width = selectVideo.videoWidth;
    selectCanvas.height = selectVideo.videoHeight;
    selectCtx.drawImage(selectVideo, 0, 0);
    document.getElementById('confirmSelect').style.display = 'inline';
});

document.getElementById('confirmSelect').addEventListener('click', async () => {
    if (selectCanvas.width === 0 || selectCanvas.height === 0) {
        alert('Imagem não capturada. Tente novamente.');
        return;
    }
    const predictions = await model.detect(selectCanvas);
    if (predictions.length === 0) {
        alert('Nenhum objeto detectado na imagem. Tente capturar novamente.');
        return;
    }
    referenceClass = predictions[0].class;
    const [x, y, width, height] = predictions[0].bbox;
    if (width <= 0 || height <= 0) {
        alert('Objeto inválido detectado. Tente outro.');
        return;
    }
    const cropped = document.createElement('canvas');
    const croppedCtx = cropped.getContext('2d');
    cropped.width = Math.max(1, width);
    cropped.height = Math.max(1, height);
    croppedCtx.drawImage(selectCanvas, x, y, width, height, 0, 0, width, height);
    
    try {
        const img = tf.browser.fromPixels(cropped);
        const resized = tf.image.resizeBilinear(img, [224, 224]);
        const normalized = resized.div(255);
        const batched = normalized.expandDims(0);
        referenceFeatures = mobilenetModel.infer(batched, true).squeeze();
        
        document.getElementById('selectMode').style.display = 'none';
        document.getElementById('searchMode').style.display = 'block';
        document.getElementById('status').textContent = 'Preparando busca... Aguarde 3 segundos.';
        setTimeout(() => {
            startSearch();
        }, 3000);
    } catch (error) {
        console.error('Error extracting features:', error);
        alert('Erro ao processar objeto. Tente novamente.');
    }
});

async function startSearch() {
    console.log('Starting search...');
    try {
        const facingMode = document.getElementById('searchCamera').value;
        const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode } });
        searchVideo.srcObject = stream;
        console.log('Video stream set');
        // Wait for video to load and have dimensions
        await new Promise(resolve => {
            searchVideo.onloadedmetadata = () => {
                console.log('Video metadata loaded:', searchVideo.videoWidth, searchVideo.videoHeight);
                if (searchVideo.videoWidth > 0 && searchVideo.videoHeight > 0) resolve();
            };
        });
        isSearching = true;
        searchLoop();
    } catch (error) {
        console.error('Error starting search:', error);
        alert('Erro ao iniciar câmera de busca.');
    }
}

async function searchLoop() {
    if (!isSearching) return;
    console.log('Search loop running, video dimensions:', searchVideo.videoWidth, searchVideo.videoHeight);
    if (searchVideo.videoWidth === 0 || searchVideo.videoHeight === 0) {
        console.log('Video not ready, retrying...');
        setTimeout(searchLoop, 100);
        return;
    }
    try {
        searchCanvas.width = searchVideo.videoWidth;
        searchCanvas.height = searchVideo.videoHeight;
        console.log('Canvas set to:', searchCanvas.width, searchCanvas.height);
        searchCtx.drawImage(searchVideo, 0, 0);
        
        const predictions = await model.detect(searchCanvas);
        console.log('Predictions:', predictions.length);
        let found = false;
        let maxSimilarity = 0;
        for (const prediction of predictions) {
            const [x, y, width, height] = prediction.bbox;
            console.log('Processing prediction:', prediction.class, x, y, width, height);
            if (width <= 0 || height <= 0 || x < 0 || y < 0) {
                console.log('Skipping invalid bbox');
                continue;
            }
            const cropped = document.createElement('canvas');
            const croppedCtx = cropped.getContext('2d');
            cropped.width = Math.max(1, width);
            cropped.height = Math.max(1, height);
            croppedCtx.drawImage(searchCanvas, x, y, width, height, 0, 0, width, height);
            
            try {
                const img = tf.browser.fromPixels(cropped);
                const resized = tf.image.resizeBilinear(img, [224, 224]);
                const normalized = resized.div(255);
                const batched = normalized.expandDims(0);
                const features = mobilenetModel.infer(batched, true).squeeze();
                
                const similarity = cosineSimilarity(referenceFeatures, features);
                console.log(`Similarity for ${prediction.class}: ${similarity}`);
                maxSimilarity = Math.max(maxSimilarity, similarity);
                if (similarity > 0.5) {
                    found = true;
                    searchCtx.strokeStyle = 'red';
                    searchCtx.lineWidth = 4;
                    searchCtx.strokeRect(x, y, width, height);
                    document.getElementById('status').textContent = `Objeto encontrado: ${prediction.class}! Similaridade: ${(similarity * 100).toFixed(1)}%`;
                    setTimeout(() => {
                        isSearching = false;
                        alert(`Objeto localizado: ${prediction.class}!`);
                    }, 2000);
                    break;
                }
            } catch (e) {
                console.error('Error processing prediction:', e);
                continue;
            }
        }
        if (!found) {
            document.getElementById('status').textContent = `Procurando... (máx similaridade: ${(maxSimilarity * 100).toFixed(1)}%)`;
            setTimeout(searchLoop, 1000);
        }
    } catch (error) {
        console.error('Error in searchLoop:', error);
        setTimeout(searchLoop, 1000);
    }
}

function cosineSimilarity(a, b) {
    const dot = tf.dot(a, b).dataSync()[0];
    const normA = tf.norm(a).dataSync()[0];
    const normB = tf.norm(b).dataSync()[0];
    return dot / (normA * normB);
}

document.getElementById('reset').addEventListener('click', () => {
    isSearching = false;
    if (selectVideo.srcObject) selectVideo.srcObject.getTracks().forEach(t => t.stop());
    if (searchVideo.srcObject) searchVideo.srcObject.getTracks().forEach(t => t.stop());
    referenceFeatures = null;
    referenceClass = null;
    document.getElementById('selectMode').style.display = 'block';
    document.getElementById('searchMode').style.display = 'none';
    document.getElementById('captureSelect').style.display = 'none';
    document.getElementById('confirmSelect').style.display = 'none';
});

loadModels();