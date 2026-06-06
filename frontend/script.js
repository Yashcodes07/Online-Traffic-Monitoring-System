// -------------------------------------------------------
// CONFIG — change this to your deployed backend URL
// For local testing: http://127.0.0.1:5000/api
// For deployed:      https://your-backend.onrender.com/api
// -------------------------------------------------------
const API_URL = window.location.hostname === '127.0.0.1' || window.location.hostname === 'localhost'
    ? 'http://127.0.0.1:5000/api'
    : '/api';  // same origin on deployment

const MAX_SIZE_MB = 100;

// DOM
const uploadArea    = document.getElementById('uploadArea');
const fileInput     = document.getElementById('fileInput');
const detectBtn     = document.getElementById('detectBtn');
const uploadContent = document.getElementById('uploadContent');
const previewContent= document.getElementById('previewContent');
const previewImage  = document.getElementById('previewImage');
const previewVideo  = document.getElementById('previewVideo');
const previewFileName = document.getElementById('previewFileName');
const resultsSection= document.getElementById('resultsSection');
const imageResults  = document.getElementById('imageResults');
const videoResults  = document.getElementById('videoResults');
const originalImage = document.getElementById('originalImage');
const detectedImage = document.getElementById('detectedImage');
const resultVideo   = document.getElementById('resultVideo');
const downloadLink  = document.getElementById('downloadLink');
const loadingOverlay= document.getElementById('loadingOverlay');
const loadingText   = document.getElementById('loadingText');
const statusEl      = document.getElementById('status');
const totalCount    = document.getElementById('totalCount');
const helmetCount   = document.getElementById('helmetCount');
const noHelmetCount = document.getElementById('noHelmetCount');
const complianceRate= document.getElementById('complianceRate');

let selectedFile = null;

// -------------------------------------------------------
// INIT
// -------------------------------------------------------
document.addEventListener('DOMContentLoaded', () => {
    checkServerHealth();
    setupEventListeners();
});

// -------------------------------------------------------
// SERVER HEALTH
// -------------------------------------------------------
async function checkServerHealth() {
    try {
        const res = await fetch(`${API_URL}/health`, { signal: AbortSignal.timeout(5000) });
        if (res.ok) setStatus('Ready', 'success');
        else setStatus('Server Error', 'error');
    } catch {
        setStatus('Server Offline', 'error');
    }
}

function setStatus(text, type) {
    const dot  = statusEl.querySelector('.status-dot');
    const span = statusEl.querySelector('.status-text');
    span.textContent = text;
    const colors = { success: '#10b981', error: '#ef4444', processing: '#3b82f6' };
    dot.style.background = colors[type] || colors.success;
}

// -------------------------------------------------------
// EVENT LISTENERS
// -------------------------------------------------------
function setupEventListeners() {
    uploadArea.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', e => { if (e.target.files[0]) handleFile(e.target.files[0]); });
    uploadArea.addEventListener('dragover',  e => { e.preventDefault(); uploadArea.classList.add('dragover'); });
    uploadArea.addEventListener('dragleave', e => { e.preventDefault(); uploadArea.classList.remove('dragover'); });
    uploadArea.addEventListener('drop', e => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
    });
    detectBtn.addEventListener('click', runDetection);
}

// -------------------------------------------------------
// FILE HANDLING
// -------------------------------------------------------
function handleFile(file) {
    // Size check
    if (file.size > MAX_SIZE_MB * 1024 * 1024) {
        alert(`File too large. Maximum allowed size is ${MAX_SIZE_MB}MB.`);
        return;
    }

    const isImage = file.type.startsWith('image/');
    const isVideo = file.type.startsWith('video/');

    if (!isImage && !isVideo) {
        alert('Unsupported file type. Please upload an image (JPG, PNG, BMP) or video (MP4, AVI, MOV).');
        return;
    }

    selectedFile = file;
    previewFileName.textContent = file.name;

    // Show preview
    uploadContent.style.display = 'none';
    previewContent.style.display = 'flex';
    previewImage.style.display = 'none';
    previewVideo.style.display = 'none';

    const url = URL.createObjectURL(file);

    if (isImage) {
        previewImage.src = url;
        previewImage.style.display = 'block';
        originalImage.src = url;
    } else {
        previewVideo.src = url;
        previewVideo.style.display = 'block';
    }

    detectBtn.disabled = false;
    resultsSection.classList.remove('active');
}

// -------------------------------------------------------
// DETECTION
// -------------------------------------------------------
async function runDetection() {
    if (!selectedFile) return;

    const isVideo = selectedFile.type.startsWith('video/');
    showLoading(true, isVideo ? 'Processing video — this may take a while...' : 'Analyzing image...');
    setStatus('Processing', 'processing');
    detectBtn.disabled = true;

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
        const res = await fetch(`${API_URL}/predict`, { method: 'POST', body: formData });

        if (!res.ok) {
            const err = await res.json().catch(() => ({ error: 'Unknown error' }));
            throw new Error(err.error || 'Detection failed');
        }

        const data = await res.json();

        if (!data.success) throw new Error(data.error || 'Detection failed');

        showResults(data);
        setStatus('Complete', 'success');

    } catch (err) {
        alert(`Error: ${err.message}`);
        setStatus('Error', 'error');
    } finally {
        showLoading(false);
        detectBtn.disabled = false;
    }
}

// -------------------------------------------------------
// SHOW RESULTS
// -------------------------------------------------------
function showResults(data) {
    imageResults.style.display = 'none';
    videoResults.style.display = 'none';

    if (data.type === 'image') {
        imageResults.style.display = 'block';
        detectedImage.src = data.image;

    } else if (data.type === 'video') {
        videoResults.style.display = 'block';
        const videoUrl = `${API_URL.replace('/api', '')}${data.download_url}`;
        resultVideo.src = videoUrl;
        downloadLink.href = videoUrl;
    }

    // Stats
    const counts = data.counts || {};
    const h  = counts.helmet    || 0;
    const nh = counts.no_helmet || 0;
    const total = h + nh;
    const compliance = data.compliance ?? (total > 0 ? Math.round(h / total * 100) : 0);

    animateValue(totalCount,     0, total,      800);
    animateValue(helmetCount,    0, h,           800);
    animateValue(noHelmetCount,  0, nh,          800);
    animateValue(complianceRate, 0, compliance,  800, '%');

    resultsSection.classList.add('active');
}

// -------------------------------------------------------
// HELPERS
// -------------------------------------------------------
function animateValue(el, start, end, duration, suffix = '') {
    const step = (end - start) / (duration / 16);
    let cur = start;
    const timer = setInterval(() => {
        cur += step;
        if ((step > 0 && cur >= end) || (step < 0 && cur <= end) || step === 0) {
            cur = end;
            clearInterval(timer);
        }
        el.textContent = Math.round(cur) + suffix;
    }, 16);
}

function showLoading(show, message = 'Analyzing...') {
    loadingText.textContent = message;
    loadingOverlay.classList.toggle('active', show);
}