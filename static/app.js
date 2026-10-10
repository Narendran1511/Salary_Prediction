/**
 * AI Salary Predictor Dashboard Logic
 * IBM BOB Internship Project / Bharat Cares
 */

// Application State
const state = {
    age: 25,
    experience: 1.0,
    gender: 'Male',
    education: "Bachelor's",
    jobTitle: 'Data Analyst',
    currentSalary: 37317,
    theme: 'sunset'
};

// DOM References
const elements = {
    // Inputs
    ageInput: document.getElementById('ageInput'),
    ageSlider: document.getElementById('ageSlider'),
    ageBadge: document.getElementById('ageBadge'),
    expInput: document.getElementById('expInput'),
    expSlider: document.getElementById('expSlider'),
    expBadge: document.getElementById('expBadge'),
    genderInput: document.getElementById('genderInput'),
    eduInput: document.getElementById('eduInput'),
    jobTitleSelect: document.getElementById('jobTitleSelect'),
    domainMeta: document.getElementById('domainMeta'),
    predictBtn: document.getElementById('predictBtn'),
    btnSpinner: document.getElementById('btnSpinner'),

    // Result Card
    salaryNumber: document.getElementById('salaryNumber'),
    bandLower: document.getElementById('bandLower'),
    bandUpper: document.getElementById('bandUpper'),
    bandFill: document.getElementById('bandFill'),
    monthlyVal: document.getElementById('monthlyVal'),
    hourlyVal: document.getElementById('hourlyVal'),
    percentileVal: document.getElementById('percentileVal'),

    // Right Column Analytics
    trajectorySvg: document.getElementById('trajectorySvg'),
    trajectoryPath: document.getElementById('trajectoryPath'),
    trajectoryArea: document.getElementById('trajectoryArea'),
    trajectoryDots: document.getElementById('trajectoryDots'),
    chartTooltip: document.getElementById('chartTooltip'),
    milestonesRow: document.getElementById('milestonesRow'),

    // Degree ROI
    roiBachelorsVal: document.getElementById('roiBachelorsVal'),
    roiMastersVal: document.getElementById('roiMastersVal'),
    roiMastersDiff: document.getElementById('roiMastersDiff'),
    roiPhdVal: document.getElementById('roiPhdVal'),
    roiPhdDiff: document.getElementById('roiPhdDiff'),

    // Market Gauge
    gaugeMarker: document.getElementById('gaugeMarker'),
    gaugeBubble: document.getElementById('gaugeBubble'),
    gaugeMetaText: document.getElementById('gaugeMetaText'),

    // Theme Toggle
    themeToggleBtn: document.getElementById('themeToggleBtn'),
    themeLabel: document.getElementById('themeLabel'),

    // Personas
    personaChips: document.getElementById('personaChips')
};

// Preset Personas
const PERSONAS = {
    'entry-analyst': {
        age: 25,
        experience: 1.0,
        gender: 'Male',
        education: "Bachelor's",
        jobTitle: 'Data Analyst'
    },
    'software-dev': {
        age: 30,
        experience: 4.0,
        gender: 'Male',
        education: "Master's",
        jobTitle: 'Software Engineer'
    },
    'senior-data': {
        age: 38,
        experience: 10.0,
        gender: 'Female',
        education: 'PhD',
        jobTitle: 'Senior Data Scientist'
    },
    'director-ops': {
        age: 48,
        experience: 18.0,
        gender: 'Female',
        education: "Master's",
        jobTitle: 'Director of Operations'
    }
};

// Department & Seniority Heuristics for metadata display
function inferMetadata(title) {
    const t = String(title).toLowerCase();
    let seniority = "Mid-Level";
    if (t.includes('ceo') || t.includes('cto') || t.includes('vp') || t.includes('director') || t.includes('chief')) {
        seniority = "Executive Tier";
    } else if (t.includes('senior') || t.includes('principal') || t.includes('lead')) {
        seniority = "Senior Professional";
    } else if (t.includes('junior') || t.includes('entry') || t.includes('assistant') || t.includes('coordinator')) {
        seniority = "Junior / Entry";
    }

    let dept = "Business & Strategy";
    if (t.includes('data') || t.includes('analyst') || t.includes('scientist') || t.includes('bi')) {
        dept = "Data & Analytics";
    } else if (t.includes('software') || t.includes('engineer') || t.includes('developer') || t.includes('it') || t.includes('tech')) {
        dept = "Engineering & IT";
    } else if (t.includes('marketing') || t.includes('sales') || t.includes('social') || t.includes('content')) {
        dept = "Sales & Marketing";
    } else if (t.includes('operations') || t.includes('supply chain') || t.includes('operations manager')) {
        dept = "Operations & Management";
    } else if (t.includes('hr') || t.includes('recruiter') || t.includes('talent')) {
        dept = "Human Resources";
    } else if (t.includes('finance') || t.includes('accounting') || t.includes('advisor')) {
        dept = "Finance & Accounting";
    }

    return `${dept} • ${seniority}`;
}

// Local Fallback Estimator calibrated to Random Forest regression parameters
function localFallbackPredict(age, exp, edu, title) {
    const t = String(title).toLowerCase();
    let base = 32000;

    // Seniority lift
    if (t.includes('director') || t.includes('vp') || t.includes('chief') || t.includes('ceo') || t.includes('cto')) {
        base += 48000;
    } else if (t.includes('senior') || t.includes('principal') || t.includes('lead')) {
        base += 18000;
    } else if (t.includes('junior') || t.includes('assistant') || t.includes('coordinator')) {
        base -= 4000;
    }

    // Role domain lift
    if (t.includes('data scientist') || t.includes('software') || t.includes('architect')) {
        base += 12000;
    } else if (t.includes('product manager') || t.includes('operations')) {
        base += 8000;
    }

    // Experience effect (non-linear diminishing curve)
    base += exp * 5400 - (exp > 12 ? (exp - 12) * 1100 : 0);

    // Age tenure factor
    base += (age - 22) * 450;

    // Education
    if (edu === "Master's") base += 14000;
    if (edu === "PhD") base += 28000;

    return Math.max(30000, Math.min(250000, Math.round(base)));
}

// Smooth Rolling Counter Animation
function animateSalaryCounter(targetSalary, duration = 650) {
    const startVal = state.currentSalary;
    const endVal = targetSalary;
    const startTime = performance.now();

    function updateCounter(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        // Ease Out Cubic
        const ease = 1 - Math.pow(1 - progress, 3);
        const current = Math.round(startVal + (endVal - startVal) * ease);

        elements.salaryNumber.textContent = current.toLocaleString('en-US');

        if (progress < 1) {
            requestAnimationFrame(updateCounter);
        } else {
            elements.salaryNumber.textContent = endVal.toLocaleString('en-US');
            state.currentSalary = endVal;
        }
    }

    requestAnimationFrame(updateCounter);
}

// Update Result Card UI Details
function updateResultDetails(salary, percentile = null) {
    const lower = Math.round(salary * 0.92);
    const upper = Math.round(salary * 1.08);

    elements.bandLower.textContent = `$${lower.toLocaleString('en-US')}`;
    elements.bandUpper.textContent = `$${upper.toLocaleString('en-US')}`;

    // Fill width relative to max $250k
    const fillPercent = Math.min(100, Math.max(12, Math.round((salary / 220000) * 100)));
    elements.bandFill.style.width = `${fillPercent}%`;

    // Monthly & Hourly
    const monthly = Math.round(salary / 12);
    const hourly = (salary / 2080).toFixed(2);
    elements.monthlyVal.textContent = `$${monthly.toLocaleString('en-US')} / mo`;
    elements.hourlyVal.textContent = `$${hourly} / hr`;

    // Percentile
    let pct = percentile;
    if (pct === null) {
        if (salary <= 55000) pct = Math.round((salary / 55000) * 25);
        else if (salary <= 95000) pct = Math.round(25 + ((salary - 55000) / 40000) * 25);
        else if (salary <= 140000) pct = Math.round(50 + ((salary - 95000) / 45000) * 25);
        else pct = Math.min(99, Math.round(75 + ((salary - 140000) / 80000) * 24));
    }
    elements.percentileVal.textContent = `${pct}th Percentile`;

    // Market Gauge Position
    const gaugePct = Math.min(96, Math.max(4, Math.round((salary - 30000) / (220000 - 30000) * 100)));
    elements.gaugeMarker.style.left = `${gaugePct}%`;
    elements.gaugeBubble.textContent = `$${(salary / 1000).toFixed(1)}k`;

    let tierDesc = "Mid-Level Professional";
    if (salary < 50000) tierDesc = "Entry / Associate Band";
    else if (salary >= 150000) tierDesc = "Senior Leadership / Executive Tier";
    else if (salary >= 100000) tierDesc = "Upper-Mid Senior Band";
    elements.gaugeMetaText.innerHTML = `This profile sits in the <strong>${tierDesc}</strong> (approx. <strong>${pct}th percentile</strong> of workforce).`;
}

// Render Trajectory SVG Chart
function renderTrajectoryChart(trajectory) {
    if (!trajectory || trajectory.length === 0) return;

    const width = 500;
    const height = 200;
    const padX = 40;
    const padY = 30;

    const maxExp = 20;
    const minSalary = Math.min(...trajectory.map(t => t.projected_salary)) * 0.85;
    const maxSalary = Math.max(...trajectory.map(t => t.projected_salary)) * 1.15;

    function getCoords(exp, sal) {
        const x = padX + (exp / maxExp) * (width - padX - 20);
        const y = height - padY - ((sal - minSalary) / (maxSalary - minSalary)) * (height - 2 * padY);
        return { x, y };
    }

    const points = trajectory.map(d => getCoords(d.experience, d.projected_salary));

    // Construct SVG Line Path
    let dLine = `M ${points[0].x} ${points[0].y}`;
    for (let i = 1; i < points.length; i++) {
        // Smooth bezier curve control points
        const prev = points[i - 1];
        const curr = points[i];
        const cpX = (prev.x + curr.x) / 2;
        dLine += ` C ${cpX} ${prev.y}, ${cpX} ${curr.y}, ${curr.x} ${curr.y}`;
    }

    // Construct Area Path
    const dArea = `${dLine} L ${points[points.length - 1].x} ${height - padY} L ${points[0].x} ${height - padY} Z`;

    elements.trajectoryPath.setAttribute('d', dLine);
    elements.trajectoryArea.setAttribute('d', dArea);

    // Draw interactive Dots
    elements.trajectoryDots.innerHTML = '';
    points.forEach((pt, idx) => {
        const data = trajectory[idx];
        const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        circle.setAttribute('cx', pt.x);
        circle.setAttribute('cy', pt.y);
        circle.setAttribute('r', data.experience === state.experience ? '6' : '4');
        circle.setAttribute('fill', data.experience === state.experience ? '#8B1E2B' : '#FFF');
        circle.setAttribute('stroke', '#8B1E2B');
        circle.setAttribute('stroke-width', '2.5');
        circle.classList.add('chart-dot');

        // Hover tooltip
        circle.addEventListener('mouseenter', (e) => {
            elements.chartTooltip.style.opacity = '1';
            elements.chartTooltip.style.left = `${pt.x / width * 100}%`;
            elements.chartTooltip.style.top = `${pt.y}px`;
            elements.chartTooltip.textContent = `${data.experience} yrs exp: $${data.projected_salary.toLocaleString('en-US')}`;
        });

        circle.addEventListener('mouseleave', () => {
            elements.chartTooltip.style.opacity = '0';
        });

        elements.trajectoryDots.appendChild(circle);
    });

    // Populate Key Milestone Badges
    const milestones = [
        { label: 'Entry (1y)', data: trajectory.find(t => t.experience === 1) || trajectory[0] },
        { label: 'Mid (5y)', data: trajectory.find(t => t.experience === 5) || trajectory[2] },
        { label: 'Senior (10y)', data: trajectory.find(t => t.experience === 10) || trajectory[4] },
        { label: 'Veteran (18y)', data: trajectory.find(t => t.experience === 18) || trajectory[trajectory.length - 1] }
    ];

    elements.milestonesRow.innerHTML = milestones.map(m => `
        <div class="milestone-item">
            <span class="m-year">${m.label}</span>
            <span class="m-salary">$${(m.data.projected_salary / 1000).toFixed(0)}k</span>
        </div>
    `).join('');
}

// Update Degree ROI Comparison Cards
function updateDegreeRoi(comparisons, baseSalary) {
    if (!comparisons) return;

    const bVal = comparisons["Bachelor's"] || baseSalary;
    const mVal = comparisons["Master's"] || Math.round(bVal * 1.25);
    const pVal = comparisons["PhD"] || Math.round(bVal * 1.6);

    elements.roiBachelorsVal.textContent = `$${bVal.toLocaleString('en-US')}`;
    elements.roiMastersVal.textContent = `$${mVal.toLocaleString('en-US')}`;
    elements.roiPhdVal.textContent = `$${pVal.toLocaleString('en-US')}`;

    const mDiff = mVal - bVal;
    const mPct = Math.round((mDiff / bVal) * 100);
    elements.roiMastersDiff.textContent = `+ $${mDiff.toLocaleString('en-US')} (+${mPct}%)`;

    const pDiff = pVal - bVal;
    const pPct = Math.round((pDiff / bVal) * 100);
    elements.roiPhdDiff.textContent = `+ $${pDiff.toLocaleString('en-US')} (+${pPct}%)`;
}

// Core Execution: Fetch Prediction from Backend or Fallback
async function executePrediction() {
    elements.domainMeta.textContent = inferMetadata(state.jobTitle);

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                age: state.age,
                gender: state.gender,
                education_level: state.education,
                job_title: state.jobTitle,
                years_of_experience: state.experience
            })
        });

        if (!response.ok) throw new Error('API server unavailable');

        const result = await response.json();
        if (result.success) {
            const pred = result.prediction["Predicted Salary"];
            animateSalaryCounter(pred);
            updateResultDetails(pred, result.metrics.market_percentile);
            renderTrajectoryChart(result.trajectory);
            updateDegreeRoi(result.degree_comparison, pred);
            return;
        }
    } catch (err) {
        // Offline / Fallback mode
        console.info('Using client-side calibrated ML fallback estimator.');
        const fallbackSal = localFallbackPredict(state.age, state.experience, state.education, state.jobTitle);
        animateSalaryCounter(fallbackSal);
        updateResultDetails(fallbackSal);

        // Simulated trajectory for offline
        const simTrajectory = [0, 1, 3, 5, 8, 10, 15, 20].map(exp => ({
            experience: exp,
            projected_salary: localFallbackPredict(state.age - state.experience + exp, exp, state.education, state.jobTitle)
        }));
        renderTrajectoryChart(simTrajectory);

        const simComparisons = {
            "Bachelor's": localFallbackPredict(state.age, state.experience, "Bachelor's", state.jobTitle),
            "Master's": localFallbackPredict(state.age, state.experience, "Master's", state.jobTitle),
            "PhD": localFallbackPredict(state.age, state.experience, "PhD", state.jobTitle)
        };
        updateDegreeRoi(simComparisons, fallbackSal);
    }
}

// Bind UI Listeners & Dual Inputs
function initializeListeners() {
    // Age Input & Slider sync
    function setAge(val) {
        const v = Math.min(65, Math.max(18, Number(val) || 18));
        state.age = v;
        elements.ageInput.value = v;
        elements.ageSlider.value = v;
        elements.ageBadge.textContent = `${v} yrs`;
        executePrediction();
    }
    elements.ageInput.addEventListener('input', (e) => setAge(e.target.value));
    elements.ageSlider.addEventListener('input', (e) => setAge(e.target.value));

    // Experience Input & Slider sync
    function setExp(val) {
        const v = Math.min(35, Math.max(0, Number(val) || 0));
        state.experience = v;
        elements.expInput.value = v;
        elements.expSlider.value = v;
        elements.expBadge.textContent = `${v.toFixed(1)} yrs`;
        executePrediction();
    }
    elements.expInput.addEventListener('input', (e) => setExp(e.target.value));
    elements.expSlider.addEventListener('input', (e) => setExp(e.target.value));

    // Gender Segmented Buttons
    document.querySelectorAll('[data-gender]').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('[data-gender]').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            state.gender = btn.dataset.gender;
            elements.genderInput.value = state.gender;
            executePrediction();
        });
    });

    // Education Segmented Buttons
    document.querySelectorAll('[data-edu]').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('[data-edu]').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            state.education = btn.dataset.edu;
            elements.eduInput.value = state.education;
            executePrediction();
        });
    });

    // Job Title Select
    elements.jobTitleSelect.addEventListener('change', (e) => {
        state.jobTitle = e.target.value;
        executePrediction();
    });

    // Quick Role Chips
    document.querySelectorAll('.role-chip').forEach(chip => {
        chip.addEventListener('click', () => {
            const role = chip.dataset.title;
            state.jobTitle = role;
            elements.jobTitleSelect.value = role;
            executePrediction();
        });
    });

    // Predict Button Form Submit
    elements.predictBtn.addEventListener('click', (e) => {
        e.preventDefault();
        executePrediction();
    });

    // Quick Persona Presets
    document.querySelectorAll('.persona-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.persona-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            const p = PERSONAS[btn.dataset.persona];
            if (p) {
                state.age = p.age;
                state.experience = p.experience;
                state.gender = p.gender;
                state.education = p.education;
                state.jobTitle = p.jobTitle;

                elements.ageInput.value = p.age;
                elements.ageSlider.value = p.age;
                elements.ageBadge.textContent = `${p.age} yrs`;

                elements.expInput.value = p.experience;
                elements.expSlider.value = p.experience;
                elements.expBadge.textContent = `${p.experience} yr`;

                // Set Gender Segment
                document.querySelectorAll('[data-gender]').forEach(b => {
                    b.classList.toggle('active', b.dataset.gender === p.gender);
                });

                // Set Edu Segment
                document.querySelectorAll('[data-edu]').forEach(b => {
                    b.classList.toggle('active', b.dataset.edu === p.education);
                });

                elements.jobTitleSelect.value = p.jobTitle;
                executePrediction();
            }
        });
    });

    // Theme Switcher Toggle
    elements.themeToggleBtn.addEventListener('click', () => {
        const currentTheme = document.body.getAttribute('data-theme') || 'sunset';
        const newTheme = currentTheme === 'sunset' ? 'dark' : 'sunset';
        document.body.setAttribute('data-theme', newTheme);
        elements.themeLabel.textContent = newTheme === 'sunset' ? 'Sunset Glow' : 'Dark Obsidian';
        localStorage.setItem('salary_app_theme', newTheme);
    });

    // Load saved theme preference
    const savedTheme = localStorage.getItem('salary_app_theme');
    if (savedTheme) {
        document.body.setAttribute('data-theme', savedTheme);
        elements.themeLabel.textContent = savedTheme === 'sunset' ? 'Sunset Glow' : 'Dark Obsidian';
    }
}

// Load extra job titles into select from backend if available
async function loadJobTitles() {
    try {
        const res = await fetch('/api/analytics');
        if (!res.ok) return;
        const data = await res.json();
        if (data.success && data.job_titles) {
            const currentSelected = elements.jobTitleSelect.value;
            const existingOptions = Array.from(elements.jobTitleSelect.options).map(o => o.value);
            data.job_titles.forEach(title => {
                if (!existingOptions.includes(title)) {
                    const opt = document.createElement('option');
                    opt.value = title;
                    opt.textContent = title;
                    elements.jobTitleSelect.appendChild(opt);
                }
            });
            elements.jobTitleSelect.value = currentSelected;
        }
    } catch (e) {
        // Quiet fallback
    }
}

// Initialize Application
document.addEventListener('DOMContentLoaded', () => {
    initializeListeners();
    loadJobTitles();
    executePrediction();
});
