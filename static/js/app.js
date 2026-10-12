document.addEventListener('DOMContentLoaded', () => {
    // Tab Navigation
    const navTabs = document.querySelectorAll('.nav-tab');
    const tabContents = document.querySelectorAll('.tab-content');

    navTabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const targetTab = tab.dataset.tab;
            
            navTabs.forEach(t => t.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));

            tab.classList.add('active');
            document.getElementById(`tab-${targetTab}`).classList.add('active');

            if (targetTab === 'database') {
                loadDatabaseRecords();
            } else if (targetTab === 'analytics') {
                loadAnalytics();
            }
        });
    });

    // Subtabs in Results (Probabilities vs Report)
    const subtabBtns = document.querySelectorAll('.subtab-btn');
    subtabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const target = btn.dataset.subtab;
            subtabBtns.forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.subtab-content').forEach(c => c.classList.remove('active'));

            btn.classList.add('active');
            document.getElementById(`subtab-${target}`).classList.add('active');
        });
    });

    // Preset Samples
    const sampleButtons = document.querySelectorAll('.btn-sample');
    const sampleSelectInput = document.getElementById('sample_select');
    const patientIdInput = document.getElementById('patient_id');
    const fullNameInput = document.getElementById('full_name');
    const ageInput = document.getElementById('age');
    const genderSelect = document.getElementById('gender');
    const indicationInput = document.getElementById('clinical_indication');

    const samplePresets = {
        'sample_pneumonia': {
            id: 'PAT-4091-PN',
            name: 'Robert Hastings',
            age: 62,
            gender: 'Male',
            indication: 'Fever (38.9°C), purulent productive cough, right pleuritic chest pain',
            img: '/static/sample_images/sample_pneumonia.jpg'
        },
        'sample_cardiomegaly': {
            id: 'PAT-8823-CM',
            name: 'Margaret Jenkins',
            age: 71,
            gender: 'Female',
            indication: 'Progressive orthopnea, bilateral lower extremity pitting edema',
            img: '/static/sample_images/sample_cardiomegaly.jpg'
        },
        'sample_normal': {
            id: 'PAT-1205-NL',
            name: 'David Chen',
            age: 38,
            gender: 'Male',
            indication: 'Pre-operative routine occupational health screening',
            img: '/static/sample_images/sample_normal.jpg'
        }
    };

    sampleButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const sampleKey = btn.dataset.sample;
            const data = samplePresets[sampleKey];
            if (!data) return;

            sampleButtons.forEach(b => b.classList.remove('active-sample'));
            btn.classList.add('active-sample');

            sampleSelectInput.value = sampleKey;
            patientIdInput.value = data.id;
            fullNameInput.value = data.name;
            ageInput.value = data.age;
            genderSelect.value = data.gender;
            indicationInput.value = data.indication;

            // Show preview
            showImagePreview(data.img, `${sampleKey}.jpg`);
        });
    });

    // File Upload Drag and Drop
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('imageInput');
    const uploadIdle = document.getElementById('uploadIdle');
    const uploadPreviewBox = document.getElementById('uploadPreviewBox');
    const previewImg = document.getElementById('previewImg');
    const previewFilename = document.getElementById('previewFilename');
    const btnClearImg = document.getElementById('btnClearImg');

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            fileInput.files = e.dataTransfer.files;
            handleFileSelection(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleFileSelection(e.target.files[0]);
        }
    });

    function handleFileSelection(file) {
        sampleSelectInput.value = '';
        sampleButtons.forEach(b => b.classList.remove('active-sample'));

        if (!patientIdInput.value) {
            patientIdInput.value = 'PAT-' + Math.floor(1000 + Math.random() * 9000);
        }
        if (!fullNameInput.value) {
            fullNameInput.value = 'Patient ' + patientIdInput.value.slice(-4);
        }

        const reader = new FileReader();
        reader.onload = (event) => {
            showImagePreview(event.target.result, file.name);
        };
        reader.readAsDataURL(file);
    }

    function showImagePreview(src, filename) {
        previewImg.src = src;
        previewFilename.textContent = filename;
        uploadIdle.classList.add('hidden');
        uploadPreviewBox.classList.remove('hidden');
    }

    btnClearImg.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.value = '';
        sampleSelectInput.value = '';
        sampleButtons.forEach(b => b.classList.remove('active-sample'));
        uploadIdle.classList.remove('hidden');
        uploadPreviewBox.classList.add('hidden');
        previewImg.src = '';
    });

    // Form Submission: Scan & Localize
    const scanForm = document.getElementById('scanForm');
    const btnScan = document.getElementById('btnScan');
    const btnSpinner = document.getElementById('btnSpinner');

    scanForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        if (!fileInput.files.length && !sampleSelectInput.value) {
            alert('Please select an image file or choose one of the quick test presets!');
            return;
        }

        // Set Loading state
        btnScan.disabled = true;
        btnSpinner.classList.remove('hidden');
        btnScan.querySelector('.btn-text').textContent = 'AI Running DenseNet-121 & Grad-CAM...';

        const formData = new FormData(scanForm);

        try {
            const resp = await fetch('/api/analyze', {
                method: 'POST',
                body: formData
            });

            const data = await resp.json();

            if (!resp.ok) {
                throw new Error(data.error || 'Failed to process radiograph.');
            }

            renderScanResults(data);
            updateScansBadge();
        } catch (err) {
            alert('Analysis Error: ' + err.message);
        } finally {
            btnScan.disabled = false;
            btnSpinner.classList.add('hidden');
            btnScan.querySelector('.btn-text').textContent = '⚡ Scan & Localize Pathology';
        }
    });

    // Heatmap Opacity Slider
    const opacitySlider = document.getElementById('opacitySlider');
    const opacityVal = document.getElementById('opacityVal');
    const viewHeatmapImg = document.getElementById('viewHeatmapImg');

    opacitySlider.addEventListener('input', (e) => {
        const val = e.target.value;
        opacityVal.textContent = `${val}%`;
        viewHeatmapImg.style.opacity = (val / 100).toString();
    });

    // Render AI Scan Results
    function renderScanResults(data) {
        const emptyState = document.getElementById('emptyResultsState');
        const resultsContainer = document.getElementById('resultsContainer');
        const severityBadge = document.getElementById('severityBadge');

        emptyState.classList.add('hidden');
        resultsContainer.classList.remove('hidden');
        severityBadge.classList.remove('hidden');

        const analysis = data.analysis;

        // Primary Finding & Confidence
        document.getElementById('resPrimaryFinding').textContent = analysis.primary_finding;
        const confPct = Math.round(analysis.confidence * 100);
        document.getElementById('resConfidence').textContent = `${confPct}%`;
        document.getElementById('resConfidenceBar').style.width = `${confPct}%`;

        // Triage Badge
        document.getElementById('resTriage').textContent = analysis.severity.toUpperCase();
        severityBadge.textContent = analysis.severity.toUpperCase();
        severityBadge.className = 'severity-pill severity-' + analysis.severity.toLowerCase();

        // Image Viewers
        document.getElementById('viewOriginalImg').src = analysis.original_image_path;
        document.getElementById('viewBaseImg').src = analysis.original_image_path;
        document.getElementById('viewHeatmapImg').src = analysis.heatmap_image_path;

        // Reset opacity to 60%
        opacitySlider.value = 60;
        opacityVal.textContent = '60%';
        viewHeatmapImg.style.opacity = '0.60';

        // Render Multi-Label Probabilities
        const pathologiesGrid = document.getElementById('pathologiesGrid');
        pathologiesGrid.innerHTML = '';

        const findings = analysis.all_findings;
        const sortedEntries = Object.entries(findings).sort((a, b) => b[1] - a[1]);

        sortedEntries.forEach(([disease, prob]) => {
            const isTop = (disease === analysis.primary_finding);
            const probPct = (prob * 100).toFixed(1);

            const item = document.createElement('div');
            item.className = `path-item ${isTop ? 'primary-highlight' : ''}`;
            item.innerHTML = `
                <div class="path-meta">
                    <span class="path-name">${disease}</span>
                    <span class="path-pct">${probPct}%</span>
                </div>
                <div class="path-track">
                    <div class="path-bar" style="width: ${probPct}%"></div>
                </div>
            `;
            pathologiesGrid.appendChild(item);
        });

        // Structured Report
        document.getElementById('reportPreText').textContent = analysis.report_text;
        document.getElementById('btnPrintReport').href = `/report/${data.scan_id}`;

        // Scroll to results on mobile/small screens
        if (window.innerWidth < 1080) {
            resultsContainer.scrollIntoView({ behavior: 'smooth' });
        }
    }

    // Copy Report Text
    document.getElementById('btnCopyReport').addEventListener('click', () => {
        const text = document.getElementById('reportPreText').textContent;
        navigator.clipboard.writeText(text).then(() => {
            const btn = document.getElementById('btnCopyReport');
            const original = btn.textContent;
            btn.textContent = '✓ Copied!';
            setTimeout(() => btn.textContent = original, 2000);
        });
    });

    // Database Records Tab
    document.getElementById('btnRefreshDb').addEventListener('click', loadDatabaseRecords);

    async function loadDatabaseRecords() {
        const tbody = document.getElementById('scansTableBody');
        tbody.innerHTML = '<tr><td colspan="9" class="loading-td">Querying SQLite database...</td></tr>';

        try {
            const resp = await fetch('/api/scans');
            const data = await resp.json();
            const scans = data.scans || [];

            document.getElementById('scansCountBadge').textContent = scans.length;

            if (scans.length === 0) {
                tbody.innerHTML = '<tr><td colspan="9" class="loading-td">No scans in database yet. Run your first CXR scan in the Scan Studio tab!</td></tr>';
                return;
            }

            tbody.innerHTML = '';
            scans.forEach(s => {
                const tr = document.createElement('tr');
                const confPct = Math.round(s.confidence * 100);
                const sevClass = (s.severity || 'routine').toLowerCase();

                tr.innerHTML = `
                    <td><strong>#${s.id}</strong></td>
                    <td>${s.scan_date.slice(0, 19)}</td>
                    <td><code>${s.patient_id}</code></td>
                    <td>${s.full_name || 'Anonymous'}</td>
                    <td><span class="badge badge-info">${s.view_position || 'PA'}</span></td>
                    <td><strong>${s.primary_finding}</strong></td>
                    <td>${confPct}%</td>
                    <td><span class="severity-pill severity-${sevClass}">${s.severity}</span></td>
                    <td>
                        <div class="table-actions">
                            <button class="btn-table btn-view-scan" data-id="${s.id}">View</button>
                            <a href="/report/${s.id}" target="_blank" class="btn-table">Report</a>
                            <button class="btn-table danger btn-delete-scan" data-id="${s.id}">Del</button>
                        </div>
                    </td>
                `;
                tbody.appendChild(tr);
            });

            // Attach action listeners
            document.querySelectorAll('.btn-view-scan').forEach(b => {
                b.addEventListener('click', () => viewScanById(b.dataset.id));
            });

            document.querySelectorAll('.btn-delete-scan').forEach(b => {
                b.addEventListener('click', () => deleteScanById(b.dataset.id));
            });

        } catch (err) {
            tbody.innerHTML = `<tr><td colspan="9" class="loading-td" style="color: #ef4444;">Error loading database: ${err.message}</td></tr>`;
        }
    }

    async function viewScanById(id) {
        try {
            const resp = await fetch(`/api/scan/${id}`);
            const data = await resp.json();
            if (data.scan) {
                const s = data.scan;
                renderScanResults({
                    scan_id: s.id,
                    patient_info: {
                        patient_id: s.patient_id,
                        full_name: s.full_name,
                        age: s.age,
                        gender: s.gender,
                        view_position: s.view_position,
                        clinical_indication: s.clinical_indication
                    },
                    analysis: {
                        original_image_path: s.original_image_path,
                        heatmap_image_path: s.heatmap_image_path,
                        primary_finding: s.primary_finding,
                        confidence: s.confidence,
                        severity: s.severity,
                        all_findings: s.findings || {},
                        report_text: s.report_text
                    }
                });

                // Switch to Studio tab
                document.querySelector('[data-tab="studio"]').click();
            }
        } catch (e) {
            alert('Failed to load scan #' + id);
        }
    }

    async function deleteScanById(id) {
        if (!confirm(`Are you sure you want to delete scan #${id} from the database?`)) return;

        try {
            const resp = await fetch(`/api/delete/${id}`, { method: 'POST' });
            const data = await resp.json();
            if (data.success) {
                loadDatabaseRecords();
                updateScansBadge();
            }
        } catch (e) {
            alert('Failed to delete scan: ' + e.message);
        }
    }

    // Analytics Tab
    async function loadAnalytics() {
        try {
            const resp = await fetch('/api/stats');
            const stats = await resp.json();

            document.getElementById('statTotalScans').textContent = stats.total_scans;
            document.getElementById('statAbnormalScans').textContent = stats.abnormal_scans;
            document.getElementById('statNormalScans').textContent = stats.normal_scans;
            document.getElementById('statTotalPatients').textContent = stats.total_patients;

            const topFindingsContainer = document.getElementById('topFindingsList');
            topFindingsContainer.innerHTML = '';

            if (stats.top_findings && stats.top_findings.length) {
                stats.top_findings.forEach(f => {
                    const row = document.createElement('div');
                    row.className = 'top-finding-row';
                    row.innerHTML = `
                        <span><strong>${f.primary_finding}</strong></span>
                        <span class="badge badge-info">${f.count} cases</span>
                    `;
                    topFindingsContainer.appendChild(row);
                });
            } else {
                topFindingsContainer.innerHTML = '<p style="color: var(--text-dim); font-size: 13px;">No scans recorded yet.</p>';
            }
        } catch (e) {
            console.error('Analytics load error:', e);
        }
    }

    async function updateScansBadge() {
        try {
            const resp = await fetch('/api/scans');
            const data = await resp.json();
            document.getElementById('scansCountBadge').textContent = (data.scans || []).length;
        } catch (e) {}
    }

    // Initialize badge count on load
    updateScansBadge();
});
