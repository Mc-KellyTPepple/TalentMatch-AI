"use strict";

/* ============================================================
   TalentMatch AI — web interface
   ============================================================ */

/* Same-origin by default (the page is served by the API itself).
   To host the page elsewhere, set window.TALENTMATCH_API_BASE before this script. */
const API_BASE_URL = String(window.TALENTMATCH_API_BASE || "").replace(/\/$/, "");
const API_ENDPOINT = `${API_BASE_URL}/analyze`;
const HEALTH_ENDPOINT = `${API_BASE_URL}/health`;

const MAX_FILE_SIZE = 10 * 1024 * 1024;
const SUPPORTED_EXTENSIONS = [".pdf", ".docx", ".txt"];
const ANALYSIS_TIMEOUT_MS = 3 * 60 * 1000;

/* ============================================================
   DOM
   ============================================================ */

const resumeForm = document.getElementById("resumeForm");
const resumeInput = document.getElementById("resume");
const uploadArea = document.getElementById("uploadArea");
const selectedFile = document.getElementById("selectedFile");
const fileName = document.getElementById("fileName");
const fileSize = document.getElementById("fileSize");
const removeFile = document.getElementById("removeFile");
const analyzeButton = document.getElementById("analyzeButton");
const buttonText = document.getElementById("buttonText");
const buttonLoader = document.getElementById("buttonLoader");
const errorMessage = document.getElementById("errorMessage");
const loadingSection = document.getElementById("loadingSection");
const resultsSection = document.getElementById("resultsSection");
const newAnalysis = document.getElementById("newAnalysis");
const jobsContainer = document.getElementById("jobsContainer");
const rolesContainer = document.getElementById("rolesContainer");
const noticesContainer = document.getElementById("noticesContainer");
const interviewSection = document.getElementById("interviewSection");
const interviewContainer = document.getElementById("interviewContainer");
const countryInput = document.getElementById("country");
const remoteOnlyInput = document.getElementById("remoteOnly");
const onlyEligibleInput = document.getElementById("onlyEligible");
const useAiInput = document.getElementById("useAi");

let analysisInProgress = false;
let analysisAbortController = null;

/* ============================================================
   Utilities
   ============================================================ */

function valueToText(value) {
    if (value === null || value === undefined) return "";
    if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") return String(value);
    if (Array.isArray(value)) return value.map(valueToText).filter(Boolean).join(", ");
    if (typeof value === "object") return Object.entries(value).map(([k, v]) => {
        const t = valueToText(v);
        return t ? `${k}: ${t}` : "";
    }).filter(Boolean).join("; ");
    return String(value);
}

function valueToNumber(value) {
    if (value === null || value === undefined || typeof value === "object") return 0;
    const n = Number(value);
    return Number.isFinite(n) ? n : 0;
}

function escapeHTML(value) {
    const div = document.createElement("div");
    div.textContent = valueToText(value);
    return div.innerHTML;
}

function normalizeArray(value) {
    if (Array.isArray(value)) return value;
    if (value === null || value === undefined || value === "") return [];
    return [value];
}

function normalizeScore(value) {
    let score = valueToNumber(value);
    if (score > 0 && score <= 1) score *= 100;
    return Number(Math.max(0, Math.min(100, score)).toFixed(2));
}

function formatFileSize(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

/* ============================================================
   Backend Wake-Up
   ============================================================ */

/*
   This background request wakes a sleeping free-tier server
   when somebody opens TalentMatch AI.

   It does NOT block the page or the user interface.
*/
async function wakeBackend() {
    try {
        const response = await fetch(`${HEALTH_ENDPOINT}?t=${Date.now()}`, {
            method: "GET",
            cache: "no-store"
        });

        if (response.ok) {
            console.log("TalentMatch AI backend is awake.");
            return true;
        }

        console.warn("TalentMatch backend returned:", response.status);
        return false;
    } catch (error) {
        console.warn("TalentMatch backend wake-up failed:", error);
        return false;
    }
}

/* ============================================================
   File Validation
   ============================================================ */

function validateFile(file) {
    if (!file) return {
        valid: false,
        message: "Please select a resume."
    };

    const name = valueToText(file.name);
    const extension = name.includes(".")
        ? "." + name.split(".").pop().toLowerCase()
        : "";

    if (!SUPPORTED_EXTENSIONS.includes(extension)) return {
        valid: false,
        message: "Unsupported file format. Please upload a PDF, DOCX or TXT resume."
    };

    if (file.size <= 0) return {
        valid: false,
        message: "The selected file is empty."
    };

    if (file.size > MAX_FILE_SIZE) return {
        valid: false,
        message: "The resume is larger than the 10 MB limit."
    };

    return { valid: true, message: "" };
}

/* ============================================================
   Errors
   ============================================================ */

function showError(message) {
    if (!errorMessage) return;
    errorMessage.textContent = valueToText(message) || "An unexpected error occurred.";
    errorMessage.classList.remove("hidden");
}

function hideError() {
    if (!errorMessage) return;
    errorMessage.textContent = "";
    errorMessage.classList.add("hidden");
}

function extractErrorMessage(data) {
    if (!data) return "The server returned an empty response.";
    if (typeof data === "string") return data.trim() || "The server returned an empty response.";
    if (typeof data !== "object") return valueToText(data);

    for (const candidate of [data.error, data.detail, data.message, data.msg, data.exception]) {
        const text = valueToText(candidate);
        if (text) return text;
    }

    return "Resume analysis failed.";
}

function getHTTPErrorMessage(status) {
    switch (status) {
        case 400:
            return "The uploaded resume could not be processed. Please check that the file is a valid PDF, DOCX or TXT document.";
        case 413:
            return "The uploaded resume is too large.";
        case 429:
            return "The service is currently busy. Please wait a moment and try again.";
        case 500:
            return "TalentMatch AI encountered an internal server error while analyzing the resume.";
        case 502:
            return "The AI service connection was interrupted. Please try the analysis again.";
        case 503:
            return "TalentMatch AI is temporarily unavailable. The AI model may still be loading or the server may have run out of memory.";
        case 504:
            return "The analysis request timed out on the server. The AI model may still be initializing. Please try again.";
        default:
            return `The server returned HTTP ${status}.`;
    }
}

function formatRequestError(error) {
    if (!error) return "An unknown error occurred.";

    if (error.name === "AbortError") {
        return "The analysis is taking longer than expected. The server may be loading the AI model for the first time. Please try again.";
    }

    if (error instanceof TypeError) {
        return "Unable to connect to the TalentMatch AI server. Please check your internet connection and try again.";
    }

    return valueToText(error.message || error) || "Resume analysis failed.";
}

/* ============================================================
   Selected File
   ============================================================ */

function displaySelectedFile(file) {
    if (!file) return;

    if (fileName) fileName.textContent = valueToText(file.name);
    if (fileSize) fileSize.textContent = formatFileSize(file.size);
    if (selectedFile) selectedFile.classList.remove("hidden");
}

function clearSelectedFile() {
    if (resumeInput) resumeInput.value = "";
    if (selectedFile) selectedFile.classList.add("hidden");
    if (fileName) fileName.textContent = "";
    if (fileSize) fileSize.textContent = "";
}

if (resumeInput) {
    resumeInput.addEventListener("change", () => {
        hideError();

        const file = resumeInput.files?.[0];
        if (!file) return;

        const validation = validateFile(file);

        if (!validation.valid) {
            showError(validation.message);
            clearSelectedFile();
            return;
        }

        displaySelectedFile(file);
    });
}

if (removeFile) {
    removeFile.addEventListener("click", event => {
        event.preventDefault();
        clearSelectedFile();
        hideError();
    });
}

/* ============================================================
   Drag & Drop
   ============================================================ */

if (uploadArea) {
    uploadArea.addEventListener("dragover", event => {
        event.preventDefault();
        uploadArea.classList.add("dragging");
    });

    uploadArea.addEventListener("dragleave", () => {
        uploadArea.classList.remove("dragging");
    });

    uploadArea.addEventListener("drop", event => {
        event.preventDefault();
        uploadArea.classList.remove("dragging");
        hideError();

        const file = event.dataTransfer?.files?.[0];
        if (!file) return;

        const validation = validateFile(file);

        if (!validation.valid) {
            showError(validation.message);
            return;
        }

        try {
            const dataTransfer = new DataTransfer();
            dataTransfer.items.add(file);
            resumeInput.files = dataTransfer.files;
        } catch (error) {
            console.warn("Unable to assign dropped file.", error);
        }

        displaySelectedFile(file);
    });
}

window.addEventListener("dragover", event => event.preventDefault());
window.addEventListener("drop", event => event.preventDefault());

/* ============================================================
   Loading
   ============================================================ */

let stepTimers = [];

function setLoadingState(isLoading) {
    if (analyzeButton) analyzeButton.disabled = isLoading;
    if (buttonText) buttonText.textContent = isLoading ? "Analyzing..." : "Analyze My CV";
    if (buttonLoader) buttonLoader.classList.toggle("hidden", !isLoading);
    if (loadingSection) loadingSection.classList.toggle("hidden", !isLoading);

    stepTimers.forEach(clearTimeout);
    stepTimers = [];
    const steps = document.querySelectorAll(".loading-step");
    steps.forEach((el, i) => el.classList.toggle("active", i === 0));
    if (isLoading) {
        [1, 2].forEach(i => {
            stepTimers.push(setTimeout(() => {
                steps.forEach((el, j) => el.classList.toggle("active", j === i));
            }, i * 3500));
        });
    }
}

/* ============================================================
   Reset
   ============================================================ */

function resetResults() {
    [jobsContainer, rolesContainer, interviewContainer, noticesContainer].forEach(el => { if (el) el.innerHTML = ""; });
    if (resultsSection) resultsSection.classList.add("hidden");
    if (interviewSection) interviewSection.classList.add("hidden");
    ["learnSection", "tipsSection"].forEach(id => document.getElementById(id)?.classList.add("hidden"));
}

/* ============================================================
   API Request
   ============================================================ */

async function fetchAnalysis(formData) {
    analysisAbortController = new AbortController();
    const timeout = setTimeout(() => analysisAbortController?.abort(), ANALYSIS_TIMEOUT_MS);

    try {
        const response = await fetch(API_ENDPOINT, {
            method: "POST",
            body: formData,
            signal: analysisAbortController.signal,
            headers: { "Accept": "application/json" }
        });

        const responseText = await response.text();
        let data = null;
        try { data = responseText.trim() ? JSON.parse(responseText) : null; } catch (e) { data = null; }

        if (!response.ok) {
            const backendMessage = data ? extractErrorMessage(data) : "";
            throw new Error(backendMessage && backendMessage !== "Resume analysis failed."
                ? backendMessage : getHTTPErrorMessage(response.status));
        }
        if (!data || typeof data !== "object" || Array.isArray(data)) {
            throw new Error("The server returned an empty or invalid response. Please try again.");
        }
        if (data.success === false) throw new Error(extractErrorMessage(data));
        return data;
    } finally {
        clearTimeout(timeout);
        analysisAbortController = null;
    }
}

if (resumeForm) {
    resumeForm.addEventListener("submit", async event => {
        event.preventDefault();
        if (analysisInProgress) return;
        hideError();

        const file = resumeInput?.files?.[0];
        const validation = validateFile(file);
        if (!validation.valid) { showError(validation.message); return; }

        resetResults();
        analysisInProgress = true;
        setLoadingState(true);

        try {
            const formData = new FormData();
            formData.append("file", file);
            formData.append("country", (countryInput?.value || "").trim());
            formData.append("remote_only", remoteOnlyInput?.checked ? "true" : "false");
            formData.append("only_eligible", onlyEligibleInput?.checked ? "true" : "false");
            formData.append("use_ai", useAiInput?.checked ? "true" : "false");

            displayResults(await fetchAnalysis(formData));
        } catch (error) {
            console.error("TalentMatch AI analysis error:", error);
            showError(formatRequestError(error));
        } finally {
            analysisInProgress = false;
            setLoadingState(false);
        }
    });
}

/* ============================================================
   Rendering helpers
   ============================================================ */

function setText(id, value) {
    const el = document.getElementById(id);
    if (el) el.textContent = valueToText(value);
}

function chip(text, cls = "") {
    return `<span class="chip ${cls}">${escapeHTML(text)}</span>`;
}

function safeUrl(url) {
    const u = valueToText(url);
    return /^https:\/\//i.test(u) ? u : "";
}

function fitClass(score) {
    if (score >= 65) return "high";
    if (score >= 35) return "mid";
    return "low";
}

/* ============================================================
   Results
   ============================================================ */

function displayResults(data) {
    const summary = data.summary || {};
    const profile = data.profile || {};
    const roles = normalizeArray(data.roles);
    const jobs = normalizeArray(data.jobs);

    const meta = [];
    if (data.filename) meta.push(valueToText(data.filename));
    if (data.processing_time_seconds) meta.push(`analysed in ${data.processing_time_seconds}s`);
    setText("reportMeta", meta.join(" · ") || "Based on the CV you uploaded.");

    // Notices
    noticesContainer.innerHTML = normalizeArray(data.notices)
        .map(n => `<div class="notice">${escapeHTML(n)}</div>`).join("");

    // Summary cards
    setText("profileLevel", profile.level || "-");
    const years = valueToNumber(profile.experience_years);
    setText("profileExperience", [years ? `~${years} yrs experience` : "", profile.education].filter(Boolean).join(" · "));
    setText("bestRoleFit", `${Math.round(valueToNumber(summary.best_role_fit))}%`);
    setText("bestRoleName", summary.best_role || "No clear role match");
    setText("jobsMatched", jobs.length);
    const scanned = valueToNumber(summary.jobs_analyzed);
    setText("jobsScanned", scanned ? `of ${scanned} live listings scanned` : "live feed unavailable");

    // Skills
    const skills = normalizeArray(profile.skills);
    document.getElementById("skillChips").innerHTML = skills.length
        ? skills.map(s => chip(s)).join("")
        : '<span class="muted">No recognisable skills were found. List tools and methods by name.</span>';

    renderRoles(roles);

    // Learn next
    const learn = normalizeArray(data.learn_next);
    const learnSection = document.getElementById("learnSection");
    if (learn.length) {
        document.getElementById("learnChips").innerHTML = learn
            .map(l => chip(`${l.skill}${l.roles > 1 ? ` · ${l.roles} roles` : ""}`, "accent")).join("");
        learnSection.classList.remove("hidden");
    }

    renderJobs(jobs, summary);

    // Tips
    const tips = normalizeArray(profile.cv_tips);
    if (tips.length) {
        document.getElementById("tipsList").innerHTML = tips.map(t => `<li>${escapeHTML(t)}</li>`).join("");
        document.getElementById("tipsSection").classList.remove("hidden");
    }

    renderInterviews(normalizeArray(data.interviews));

    resultsSection?.classList.remove("hidden");
    setTimeout(() => resultsSection?.scrollIntoView({ behavior: "smooth", block: "start" }), 100);
}

/* ============================================================
   Roles
   ============================================================ */

function renderRoles(roles) {
    rolesContainer.innerHTML = "";
    if (!roles.length) {
        rolesContainer.innerHTML = `<div class="empty-state"><h3>No roles identified</h3>
            <p>Add specific skills, tools and job titles to your CV and try again.</p></div>`;
        return;
    }

    roles.forEach((role, index) => {
        const fit = Math.round(valueToNumber(role.fit));
        const links = role.search_links || {};
        const linkHTML = ["linkedin", "indeed", "google"]
            .filter(k => safeUrl(links[k]))
            .map(k => `<a class="search-link" href="${escapeHTML(safeUrl(links[k]))}" target="_blank" rel="noopener noreferrer nofollow">${k === "google" ? "Google Jobs" : k[0].toUpperCase() + k.slice(1)}</a>`)
            .join("");
        const matched = normalizeArray(role.matched_skills);
        const missing = normalizeArray(role.missing_skills);
        const next = normalizeArray(role.next_roles);

        const card = document.createElement("article");
        card.className = `job-card role-card ${index === 0 ? "top" : ""}`;
        card.innerHTML = `
            <div class="job-card-top">
                <div class="job-rank">#${index + 1}</div>
                <div class="job-title-area">
                    <h4>${escapeHTML(role.title)}</h4>
                    <span class="match-level">${escapeHTML(role.family)} · ${escapeHTML(role.match_level)}</span>
                </div>
                <div class="match-score"><strong>${fit}%</strong><span>Fit</span></div>
            </div>
            <div class="score-bar"><div class="score-fill ${fitClass(fit)}" style="width:${fit}%"></div></div>
            <p class="role-summary">${escapeHTML(role.summary)}</p>
            <div class="skill-analysis">
                ${matched.length ? `<div class="skill-group"><strong>Skills you have</strong><div class="chips">${matched.map(s => chip(s, "good")).join("")}</div></div>` : ""}
                ${missing.length ? `<div class="skill-group"><strong>Core skills to add</strong><div class="chips">${missing.map(s => chip(s, "gap")).join("")}</div></div>` : ""}
            </div>
            ${next.length ? `<div class="career-path"><strong>Where this can lead:</strong> ${next.map(escapeHTML).join(" → ")}</div>` : ""}
            ${role.level_note ? `<p class="level-note">${escapeHTML(role.level_note)}</p>` : ""}
            ${linkHTML ? `<div class="search-links"><span>Search openings:</span>${linkHTML}</div>` : ""}`;
        rolesContainer.appendChild(card);
    });
}

/* ============================================================
   Live jobs
   ============================================================ */

function renderJobs(jobs, summary) {
    jobsContainer.innerHTML = "";

    if (!jobs.length) {
        jobsContainer.innerHTML = `<div class="empty-state"><h3>No live openings matched</h3>
            <p>Use the search links on your best-fit roles above to look for current openings.</p></div>`;
        return;
    }

    const ai = summary && summary.groq_enabled;
    document.getElementById("jobsSubtitle").textContent = ai
        ? "Current vacancies, ranked by skills fit and re-checked by AI against your CV."
        : "Current vacancies, ranked by skills, job title and wording fit with your CV.";

    jobs.forEach((job, index) => {
        const score = Math.round(valueToNumber(job.match_score));
        const url = safeUrl(job.url);
        const matched = normalizeArray(job.matched_skills);
        const missing = normalizeArray(job.missing_skills);
        const metaBits = [job.company, job.location, job.remote ? "Remote" : "", job.job_type, job.salary,
            job.posted ? `Posted ${job.posted}` : ""].filter(Boolean);
        const eligibility = valueToText(job.eligibility);
        const restricted = /^(Restricted|On-site|Visa|Remote, but)/.test(eligibility);

        const card = document.createElement("article");
        card.className = "job-card";
        card.innerHTML = `
            <div class="job-card-top">
                <div class="job-rank">#${index + 1}</div>
                <div class="job-title-area">
                    <h4>${escapeHTML(job.title)}</h4>
                    <span class="match-level">${escapeHTML(job.match_level)}${job.ai_reranked ? " · AI-checked" : ""}</span>
                </div>
                <div class="match-score"><strong>${score}%</strong><span>Match</span></div>
            </div>
            <div class="score-bar"><div class="score-fill ${fitClass(score)}" style="width:${score}%"></div></div>
            <div class="job-meta">${metaBits.map(b => `<span>${escapeHTML(b)}</span>`).join("")}</div>
            ${eligibility ? `<div class="eligibility ${restricted ? "warn" : ""}">${escapeHTML(eligibility)}</div>` : ""}
            ${job.reason ? `<p class="job-reason">${escapeHTML(job.reason)}</p>` : ""}
            <div class="skill-analysis">
                ${matched.length ? `<div class="skill-group"><strong>You match</strong><div class="chips">${matched.map(s => chip(s, "good")).join("")}</div></div>` : ""}
                ${missing.length ? `<div class="skill-group"><strong>Listing also asks for</strong><div class="chips">${missing.map(s => chip(s, "gap")).join("")}</div></div>` : ""}
            </div>
            ${job.level_note ? `<p class="level-note">${escapeHTML(job.level_note)}</p>` : ""}
            <div class="job-actions">
                ${url ? `<a class="apply-button" href="${escapeHTML(url)}" target="_blank" rel="noopener noreferrer nofollow">View &amp; apply on ${escapeHTML(job.source)} ↗</a>` : ""}
            </div>`;
        jobsContainer.appendChild(card);
    });
}

/* ============================================================
   Interviews
   ============================================================ */

function renderInterviews(questions) {
    interviewContainer.innerHTML = "";
    if (!questions.length) { interviewSection?.classList.add("hidden"); return; }
    interviewSection?.classList.remove("hidden");

    questions.forEach((q, index) => {
        const card = document.createElement("article");
        card.className = "interview-card";
        card.innerHTML = `
            <div class="interview-header">
                <div class="interview-number">${index + 1}</div>
                <div class="interview-meta">
                    ${q.role ? `<span>${escapeHTML(q.role)}</span>` : ""}
                    ${q.category ? `<span>${escapeHTML(q.category)}</span>` : ""}
                    ${q.difficulty ? `<span>${escapeHTML(q.difficulty)}</span>` : ""}
                </div>
            </div>
            <div class="interview-question"><strong>Question</strong><p>${escapeHTML(q.question)}</p></div>
            ${q.answer ? `<details class="ideal-answer"><summary>How to answer</summary><p>${escapeHTML(q.answer)}</p></details>` : ""}`;
        interviewContainer.appendChild(card);
    });
}

/* ============================================================
   New Analysis
   ============================================================ */

if (newAnalysis) {
    newAnalysis.addEventListener("click", () => {
        if (analysisInProgress) return;
        resetResults();
        clearSelectedFile();
        hideError();
        window.scrollTo({ top: 0, behavior: "smooth" });
    });
}

document.addEventListener("DOMContentLoaded", () => {
    resetResults();
    wakeBackend();
});
