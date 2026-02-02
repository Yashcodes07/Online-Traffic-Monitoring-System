// API Configuration
const API_URL = 'http://127.0.0.1:5000/api';   // ✅ FIXED (was localhost)

// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const detectBtn = document.getElementById('detectBtn');
const resultsSection = document.getElementById('resultsSection');
const loadingOverlay = document.getElementById('loadingOverlay');
const originalImage = document.getElementById('originalImage');
const detectedImage = document.getElementById('detectedImage');
const statsPanel = document.getElementById('statsPanel');
const status = document.getElementById('status');

// Stats elements
const motorcycleCount = document.getElementById('motorcycleCount');
const helmetCount = document.getElementById('helmetCount');
const noHelmetCount = document.getElementById('noHelmetCount');
const complianceRate = document.getElementById('complianceRate');

let selectedFile = null;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    checkServerHealth();
    initializeEventListeners();
});

// Check if backend server is running
async function checkServerHealth() {
    try {
        const response = await fetch(`${API_URL}/health`);
        if (response.ok) {
            updateStatus('Ready', 'success');
        } else {
            updateStatus('Server Error', 'error');
        }
    } catch (error) {
        updateStatus('Server Offline', 'error');
        console.error('Server health check failed:', error);
    }
}

// Update status indicator
function updateStatus(text, type) {
    const statusText = status.querySelector('.status-text');
    const statusDot = status.querySelector('.status-dot');
    
    statusText.textContent = text;
    
    const colors = {
        success: '#10b981',
        error: '#ef4444',
        warning: '#f59e0b',
        processing: '#3b82f6'
    };
    
    statusDot.style.background = colors[type] || colors.success;
}

// Initialize event listeners
function initializeEventListeners() {
    uploadArea.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFileSelect);
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);
    detectBtn.addEventListener('click', handleDetection);
}

// Handle file selection
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) processFile(file);
}

// Drag events
function handleDragOver(e) {
    e.preventDefault();
    uploadArea.classList.add('dragover');
}
function handleDragLeave(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
}
function handleDrop(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file) processFile(file);
}

// Process selected file
function processFile(file) {
    if (!file.type.startsWith('image/')) {
        alert('Please select an image file (JPG, PNG, BMP)');
        return;
    }
    
    selectedFile = file;
    
    const reader = new FileReader();
    reader.onload = (e) => {
        originalImage.src = e.target.result;
        detectBtn.disabled = false;
    };
    reader.readAsDataURL(file);
}

// Handle detection
async function handleDetection() {
    if (!selectedFile) return;
    
    showLoading(true);
    updateStatus('Processing', 'processing');
    
    const formData = new FormData();
    formData.append('file', selectedFile);
    
    try {
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) throw new Error('Detection failed');
        
        const result = await response.json();
        
        if (result.success) {
            displayResults(result);
            updateStatus('Complete', 'success');
        } else {
            throw new Error(result.error);
        }
        
    } catch (error) {
        console.error(error);
        alert('Server not connected');
        updateStatus('Error', 'error');
    } finally {
        showLoading(false);
    }
}

// Display detection results
function displayResults(result) {
    // ✅ FIXED: backend returns `image`, not `output_image`
    if (result.type === 'image' && result.image) {
        detectedImage.src = result.image;
    }
    
    const detections = result.detections || [];
    const totalMotorcycles = detections.length;
    let withHelmet = 0;
    let withoutHelmet = 0;
    
    // ✅ FIXED: detections are strings ("Helmet", "No Helmet")
    detections.forEach(detection => {
        if (detection.toLowerCase() === 'helmet') {
            withHelmet++;
        } else {
            withoutHelmet++;
        }
    });
    
    const compliance = totalMotorcycles > 0
        ? Math.round((withHelmet / totalMotorcycles) * 100)
        : 0;
    
    updateStats(totalMotorcycles, withHelmet, withoutHelmet, compliance);
    resultsSection.classList.add('active');
}

// Update statistics
function updateStats(total, helmet, noHelmet, compliance) {
    animateValue(motorcycleCount, 0, total, 800);
    animateValue(helmetCount, 0, helmet, 800);
    animateValue(noHelmetCount, 0, noHelmet, 800);
    animateValue(complianceRate, 0, compliance, 800, '%');
}

// Animate number counting
function animateValue(element, start, end, duration, suffix = '') {
    const range = end - start;
    const increment = range / (duration / 16);
    let current = start;
    
    const timer = setInterval(() => {
        current += increment;
        if ((increment > 0 && current >= end) || (increment < 0 && current <= end)) {
            current = end;
            clearInterval(timer);
        }
        element.textContent = Math.round(current) + suffix;
    }, 16);
}

// Show/hide loading overlay
function showLoading(show) {
    loadingOverlay.classList.toggle('active', show);
}
