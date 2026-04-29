// script.js - Logic for Resume Screener Pro UI (Django Edition)

document.addEventListener('DOMContentLoaded', () => {
    // --- DOM Elements ---
    const dropZone = document.getElementById('drop-zone');
    const resumeInput = document.getElementById('resumes-upload');
    const fileCounter = document.getElementById('file-counter');
    const fileCountNumber = document.getElementById('file-count-number');
    const btnRun = document.getElementById('btn-run');
    
    const jdTextarea = document.getElementById('jd-textarea');
    const jdUpload = document.getElementById('jd-upload');
    const jdFilename = document.getElementById('jd-filename');
    
    // Views
    const viewInput = document.getElementById('view-input');
    const viewProcessing = document.getElementById('view-processing');
    const viewResults = document.getElementById('view-results');
    
    // Processing elements
    const progressBar = document.getElementById('progress-bar');
    const processingStatus = document.getElementById('processing-status');
    const processingDetail = document.getElementById('processing-detail');
    
    // Results elements
    const resultsTbody = document.getElementById('results-tbody');
    const btnNewRun = document.getElementById('btn-new-run');
    const btnExportCsv = document.getElementById('btn-export-csv');
    const btnExportJson = document.getElementById('btn-export-json');
    
    // Modal elements
    const detailModal = document.getElementById('detail-modal');
    const btnCloseModal = document.getElementById('btn-close-modal');
    
    // Settings Modal
    const settingsModal = document.getElementById('settings-modal');
    const btnOpenSettings = document.getElementById('btn-open-settings');
    const btnCloseSettings = document.getElementById('btn-close-settings');
    const nlpWeightRange = document.getElementById('nlp-weight-range');
    const aiWeightRange = document.getElementById('ai-weight-range');
    const nlpWeightVal = document.getElementById('nlp-weight-val');
    const aiWeightVal = document.getElementById('ai-weight-val');
    
    // State
    let selectedResumes = [];
    let selectedJDFile = null;
    let currentResults = [];
    
    // --- Settings Logic ---
    btnOpenSettings.addEventListener('click', () => {
        settingsModal.classList.add('active');
    });
    btnCloseSettings.addEventListener('click', () => {
        settingsModal.classList.remove('active');
    });
    settingsModal.addEventListener('click', (e) => {
        if (e.target === settingsModal) {
            settingsModal.classList.remove('active');
        }
    });
    
    nlpWeightRange.addEventListener('input', (e) => {
        let val = parseFloat(e.target.value);
        nlpWeightVal.textContent = val.toFixed(1);
        let aiVal = 1.0 - val;
        aiWeightRange.value = aiVal;
        aiWeightVal.textContent = aiVal.toFixed(1);
    });

    aiWeightRange.addEventListener('input', (e) => {
        let val = parseFloat(e.target.value);
        aiWeightVal.textContent = val.toFixed(1);
        let nlpVal = 1.0 - val;
        nlpWeightRange.value = nlpVal;
        nlpWeightVal.textContent = nlpVal.toFixed(1);
    });
    
    // --- File Handling ---
    jdUpload.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            selectedJDFile = e.target.files[0];
            jdFilename.textContent = selectedJDFile.name;
        }
    });

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });
    
    ['dragleave', 'dragend'].forEach(type => {
        dropZone.addEventListener(type, () => {
            dropZone.classList.remove('dragover');
        });
    });
    
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            handleResumeFiles(e.dataTransfer.files);
        }
    });
    
    resumeInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleResumeFiles(e.target.files);
        }
    });
    
    function handleResumeFiles(files) {
        for(let i=0; i<files.length; i++) {
            selectedResumes.push(files[i]);
        }
        fileCountNumber.textContent = selectedResumes.length;
        fileCounter.classList.remove('hidden');
    }
    
    // --- Pipeline Execution via Django ---
    btnRun.addEventListener('click', async () => {
        if (selectedResumes.length === 0) {
            alert("Please upload at least one resume to process.");
            return;
        }
        if (!selectedJDFile && !jdTextarea.value.trim()) {
            alert("Please provide a Job Description (paste text or upload file).");
            return;
        }
        
        // Show processing
        switchView(viewProcessing);
        processingStatus.textContent = "Sending batch to server...";
        processingDetail.textContent = "Starting AI pipeline execution...";
        
        // Animate pseudo progress bar
        let progress = 5;
        progressBar.style.width = `5%`;
        const progInterval = setInterval(() => {
            if (progress < 90) progress += 5;
            progressBar.style.width = `${progress}%`;
            processingStatus.textContent = "Awaiting Gemini AI scoring...";
            processingDetail.textContent = "Applying exponential backoff limits across batch... this takes 2s per resume!";
        }, 1500);
        
        // Build payload
        const formData = new FormData();
        if (selectedJDFile) {
            formData.append("jd_file", selectedJDFile);
        }
        if (jdTextarea.value.trim()) {
            formData.append("jd_text", jdTextarea.value.trim());
        }
        formData.append("nlp_weight", nlpWeightRange.value);
        formData.append("ai_weight", aiWeightRange.value);
        
        selectedResumes.forEach(file => {
            formData.append("resume_files", file);
        });
        
        try {
            const response = await fetch("/run-pipeline/", {
                method: "POST",
                body: formData
            });
            const data = await response.json();
            
            clearInterval(progInterval);
            progressBar.style.width = `100%`;
            
            if (!response.ok) {
                alert("Error from server: " + (data.error || "Unknown error"));
                switchView(viewInput);
                return;
            }
            
            currentResults = data.results;
            setTimeout(() => showResults(data.results), 500);
            
        } catch (err) {
            clearInterval(progInterval);
            alert("Network error: " + err.message);
            switchView(viewInput);
        }
    });
    
    // --- Results & Modal ---
    function switchView(viewToShow) {
        [viewInput, viewProcessing, viewResults].forEach(v => v.classList.add('hidden'));
        viewToShow.classList.remove('hidden');
        if(viewToShow === viewResults) {
            viewToShow.style.animation = 'fade-in 0.5s ease-out';
        }
    }
    
    function getScoreClass(score) {
        if (score >= 8) return 'score-high';
        if (score >= 5) return 'score-med';
        return 'score-low';
    }
    
    btnNewRun.addEventListener('click', () => {
        // Clear all state
        selectedResumes = [];
        selectedJDFile = null;
        currentResults = [];
        
        // Reset inputs
        jdFilename.textContent = "";
        jdTextarea.value = "";
        jdUpload.value = '';
        resumeInput.value = '';
        
        fileCounter.classList.add('hidden');
        fileCountNumber.textContent = "0";
        progressBar.style.width = '0%';
        switchView(viewInput);
    });
    
    function showResults(dataArray) {
        resultsTbody.innerHTML = '';
        
        dataArray.forEach(row => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>#${row.rank}</td>
                <td class="font-medium">${row.name}</td>
                <td class="text-muted">${row.nlp.toFixed(1)}</td>
                <td class="text-muted">${row.ai.toFixed(1)}</td>
                <td><span class="score-badge ${getScoreClass(row.final)}">${row.final.toFixed(1)}</span></td>
                <td><p class="text-sm text-muted truncate-2">${row.rationale}</p></td>
                <td><button class="btn btn-outline btn-sm">View <i class="fa-solid fa-arrow-right ml-2"></i></button></td>
            `;
            
            tr.addEventListener('click', () => openModal(row));
            resultsTbody.appendChild(tr);
        });
        
        switchView(viewResults);
    }
    
    // --- Export Logic ---
    btnExportJson.addEventListener('click', () => {
        if (!currentResults || currentResults.length === 0) return;
        const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(currentResults, null, 2));
        const anchor = document.createElement('a');
        anchor.href = dataStr;
        anchor.download = "screening_results.json";
        document.body.appendChild(anchor);
        anchor.click();
        anchor.remove();
    });

    btnExportCsv.addEventListener('click', () => {
        if (!currentResults || currentResults.length === 0) return;
        const headers = ["Rank", "Candidate Name", "File", "NLP Score", "AI Score", "Final Score", "Rationale"];
        let csvContent = "data:text/csv;charset=utf-8,";
        csvContent += headers.join(",") + "\r\n";
        currentResults.forEach(r => {
            let rationale = r.rationale.replace(/"/g, '""'); // Escape quotes for CSV
            let row = [r.rank, `"${r.name}"`, `"${r.file}"`, r.nlp, r.ai, r.final, `"${rationale}"`];
            csvContent += row.join(",") + "\r\n";
        });
        const anchor = document.createElement('a');
        anchor.href = encodeURI(csvContent);
        anchor.download = "screening_results.csv";
        document.body.appendChild(anchor);
        anchor.click();
        anchor.remove();
    });
    
    // Modal Logic
    function openModal(data) {
        document.getElementById('modal-candidate-name').textContent = data.name;
        document.getElementById('modal-filename').textContent = data.file;
        
        const scoreBadge = document.getElementById('modal-final-score');
        scoreBadge.textContent = `Score: ${data.final.toFixed(1)}`;
        scoreBadge.className = `badge ${getScoreClass(data.final)}`;
        
        document.getElementById('modal-rationale').textContent = data.rationale;
        
        const matchedContainer = document.getElementById('modal-matched');
        matchedContainer.innerHTML = data.matched.map(s => `<span class="tag">${s}</span>`).join('');
        if(data.matched.length === 0) matchedContainer.innerHTML = '<span class="text-sm text-muted">No key skills matched</span>';
        
        const missingContainer = document.getElementById('modal-missing');
        missingContainer.innerHTML = data.missing.map(s => `<span class="tag">${s}</span>`).join('');
        if(data.missing.length === 0) missingContainer.innerHTML = '<span class="text-sm text-muted">Nothing missing</span>';
        
        detailModal.classList.add('active');
    }
    
    btnCloseModal.addEventListener('click', () => {
        detailModal.classList.remove('active');
    });
    
    detailModal.addEventListener('click', (e) => {
        if (e.target === detailModal) {
            detailModal.classList.remove('active');
        }
    });

});
