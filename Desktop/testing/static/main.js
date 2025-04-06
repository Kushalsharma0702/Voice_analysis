document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const recordBtn = document.getElementById('recordBtn');
    const audioFileInput = document.getElementById('audioFileInput');
    const recordingStatus = document.getElementById('recordingStatus');
    const recordingTime = document.getElementById('recordingTime');
    const audioPlayback = document.getElementById('audioPlayback');
    const processingStatus = document.getElementById('processingStatus');
    const results = document.getElementById('results');
    const downloadPdfBtn = document.getElementById('downloadPdfBtn');
    
    // Variables
    let mediaRecorder;
    let audioChunks = [];
    let recordingInterval;
    let recordingSeconds = 0;
    const MAX_RECORDING_TIME = 20; // seconds
    
    // Initialize MediaRecorder if browser supports it
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        console.log('MediaRecorder is supported');
    } else {
        alert('Your browser does not support audio recording. Please try using Chrome or Firefox.');
        recordBtn.disabled = true;
    }
    
    // Recording functionality
    recordBtn.addEventListener('click', function() {
        if (mediaRecorder && mediaRecorder.state === 'recording') {
            stopRecording();
        } else {
            startRecording();
        }
    });
    
    function startRecording() {
        // Reset UI
        results.classList.add('hidden');
        audioPlayback.classList.add('hidden');
        
        // Get user media
        navigator.mediaDevices.getUserMedia({ audio: true })
            .then(stream => {
                mediaRecorder = new MediaRecorder(stream);
                audioChunks = [];
                
                mediaRecorder.ondataavailable = (event) => {
                    audioChunks.push(event.data);
                };
                
                mediaRecorder.onstop = () => {
                    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                    const audioUrl = URL.createObjectURL(audioBlob);
                    audioPlayback.src = audioUrl;
                    audioPlayback.classList.remove('hidden');
                    
                    // Process the recorded audio
                    processAudio(audioBlob);
                };
                
                // Start recording
                mediaRecorder.start();
                recordingStatus.classList.remove('hidden');
                recordBtn.innerHTML = '<i class="fas fa-stop"></i> Stop Recording';
                recordBtn.classList.add('recording');
                
                // Update recording time
                recordingSeconds = 0;
                updateRecordingTime();
                recordingInterval = setInterval(updateRecordingTime, 1000);
                
                // Auto-stop after MAX_RECORDING_TIME
                setTimeout(() => {
                    if (mediaRecorder && mediaRecorder.state === 'recording') {
                        stopRecording();
                    }
                }, MAX_RECORDING_TIME * 1000);
            })
            .catch(error => {
                console.error('Error accessing microphone:', error);
                alert('Unable to access your microphone. Please check permissions and try again.');
            });
    }
    
    function stopRecording() {
        if (mediaRecorder) {
            mediaRecorder.stop();
            clearInterval(recordingInterval);
            recordingStatus.classList.add('hidden');
            recordBtn.innerHTML = '<i class="fas fa-microphone"></i> Start Recording';
            recordBtn.classList.remove('recording');
            
            // Stop all microphone tracks
            mediaRecorder.stream.getTracks().forEach(track => track.stop());
        }
    }
    
    function updateRecordingTime() {
        recordingSeconds++;
        recordingTime.textContent = `Recording: ${recordingSeconds}s / ${MAX_RECORDING_TIME}s`;
        
        if (recordingSeconds >= MAX_RECORDING_TIME) {
            stopRecording();
        }
    }
    
    // File upload functionality
    audioFileInput.addEventListener('change', function(event) {
        const file = event.target.files[0];
        if (!file) return;
        
        // Reset UI
        results.classList.add('hidden');
        recordingStatus.classList.add('hidden');
        
        // Create audio element for playback
        const reader = new FileReader();
        reader.onload = function(e) {
            audioPlayback.src = e.target.result;
            audioPlayback.classList.remove('hidden');
        };
        reader.readAsDataURL(file);
        
        // Process the uploaded audio
        processAudio(file);
    });
    
    // Process audio (either recorded or uploaded)
    function processAudio(audioBlob) {
        processingStatus.classList.remove('hidden');
        
        // Create FormData for server upload
        const formData = new FormData();
        formData.append('file', audioBlob, 'recording.wav');
        
        // Send to server for analysis
        fetch('/analyze', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            processingStatus.classList.add('hidden');
            displayResults(data);
        })
        .catch(error => {
            console.error('Error processing audio:', error);
            processingStatus.classList.add('hidden');
            alert('An error occurred while processing your audio. Please try again.');
        });
    }
    
    // Display analysis results
    function displayResults(data) {
        if (data.error) {
            alert(`Error: ${data.error}`);
            return;
        }
        
        // Update score and confidence level
        document.getElementById('confidenceScore').textContent = `${data.confidence_score}%`;
        document.getElementById('confidenceLabel').textContent = data.confidence_level;
        
        // Update metrics
        document.getElementById('pitchMean').textContent = `${data.pitch_mean} Hz`;
        document.getElementById('pitchStd').textContent = data.pitch_std;
        document.getElementById('energyMean').textContent = data.energy_mean.toFixed(5);
        document.getElementById('pauseCount').textContent = data.pause_count;
        document.getElementById('fillerCount').textContent = data.filler_count;
        
        // Update warnings
        const warningsSection = document.getElementById('warnings');
        const warningsList = document.getElementById('warningsList');
        warningsList.innerHTML = '';
        
        if (data.warnings && data.warnings.length > 0 && data.warnings[0]) {
            data.warnings.forEach(warning => {
                const li = document.createElement('li');
                li.textContent = warning;
                warningsList.appendChild(li);
            });
            warningsSection.classList.remove('hidden');
        } else {
            warningsSection.classList.add('hidden');
        }
        
        // Update suggestions
        const suggestionsList = document.getElementById('suggestionsList');
        suggestionsList.innerHTML = '';
        
        if (data.suggestions && data.suggestions.length > 0) {
            data.suggestions.forEach(suggestion => {
                const li = document.createElement('li');
                li.textContent = suggestion;
                suggestionsList.appendChild(li);
            });
        } else {
            const li = document.createElement('li');
            li.textContent = 'Your voice sounds confident and fluent!';
            suggestionsList.appendChild(li);
        }
        
        // Update transcript if available
        const transcriptContainer = document.getElementById('transcriptContainer');
        const transcript = document.getElementById('transcript');
        
        if (data.transcribed_text && data.transcribed_text.trim() !== '') {
            transcript.textContent = data.transcribed_text;
            transcriptContainer.classList.remove('hidden');
        } else {
            transcriptContainer.classList.add('hidden');
        }
        
        // Update spectrogram
        if (data.spectrogram) {
            document.getElementById('spectrogramImage').src = '/spectrogram/' + data.spectrogram;
        }
        
        // Show results section
        results.classList.remove('hidden');
        
        // Scroll to results
        results.scrollIntoView({ behavior: 'smooth' });
    }
    
    // Download PDF functionality
    downloadPdfBtn.addEventListener('click', function() {
        const element = document.getElementById('reportContent');
        const options = {
            margin: 10,
            filename: 'VocalEdge-Report.pdf',
            image: { type: 'jpeg', quality: 0.98 },
            html2canvas: { scale: 2 },
            jsPDF: { unit: 'mm', format: 'a4', orientation: 'portrait' }
        };
        
        html2pdf().set(options).from(element).save();
    });
});