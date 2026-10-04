"use strict";

/* ============================================================
   TalentMatch AI — Vercel Frontend / Render AI Backend
   ============================================================ */

const API_BASE_URL = "https://talentmatch-ai-1-gcn6.onrender.com";
const API_ENDPOINT = `${API_BASE_URL}/analyze`;
const HEALTH_ENDPOINT = `${API_BASE_URL}/health`;

const MAX_FILE_SIZE = 10 * 1024 * 1024;
const SUPPORTED_EXTENSIONS = [".pdf", ".docx", ".txt"];
const ANALYSIS_TIMEOUT_MS = 20 * 60 * 1000;

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
const interviewSection = document.getElementById("interviewSection");
const interviewContainer = document.getElementById("interviewContainer");
const jobsAnalyzed = document.getElementById("jobsAnalyzed");
const bestMatchScore = document.getElementById("bestMatchScore");
const bestMatchLevel = document.getElementById("bestMatchLevel");
const averageMatchScore = document.getElementById("averageMatchScore");

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
   Vercel serves the frontend permanently.
   This background request wakes the Render backend when
   somebody opens TalentMatch AI.

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

function setLoadingState(isLoading) {
    if (analyzeButton) analyzeButton.disabled = isLoading;
    if (buttonText) buttonText.textContent = isLoading ? "Analyzing..." : "Analyze Resume";

    if (buttonLoader) {
        buttonLoader.classList.toggle("hidden", !isLoading);
    }

    if (loadingSection) {
        loadingSection.classList.toggle("hidden", !isLoading);
    }
}

/* ============================================================
   Reset
   ============================================================ */

function resetResults() {
    if (jobsContainer) jobsContainer.innerHTML = "";
    if (interviewContainer) interviewContainer.innerHTML = "";
    if (jobsAnalyzed) jobsAnalyzed.textContent = "0";
    if (bestMatchScore) bestMatchScore.textContent = "0%";
    if (bestMatchLevel) bestMatchLevel.textContent = "No Match";
    if (averageMatchScore) averageMatchScore.textContent = "0%";

    if (interviewSection) interviewSection.classList.add("hidden");
    if (resultsSection) resultsSection.classList.add("hidden");
}

/* ============================================================
   API Request
   ============================================================ */

async function fetchAnalysis(formData) {
    analysisAbortController = new AbortController();

    const timeout = setTimeout(() => {
        analysisAbortController?.abort();
    }, ANALYSIS_TIMEOUT_MS);

    try {
        const response = await fetch(API_ENDPOINT, {
            method: "POST",
            body: formData,
            signal: analysisAbortController.signal,
            headers: { "Accept": "application/json" }
        });

        const responseText = await response.text();

        console.log("TalentMatch HTTP status:", response.status);
        console.log("TalentMatch response length:", responseText.length);

        if (!responseText.trim()) {
            throw new Error(
                response.ok
                    ? "The server returned an empty response. The AI analysis may have stopped before producing a result."
                    : getHTTPErrorMessage(response.status)
            );
        }

        let data;

        try {
            data = JSON.parse(responseText);
        } catch (error) {
            console.error("Non-JSON server response:", responseText.substring(0, 1000));

            if (!response.ok) {
                throw new Error(getHTTPErrorMessage(response.status));
            }

            throw new Error("The server returned an invalid response instead of JSON.");
        }

        console.log("TalentMatch AI response:", data);

        if (!response.ok) {
            const backendMessage = extractErrorMessage(data);

            if (backendMessage && backendMessage !== "Resume analysis failed.") {
                throw new Error(backendMessage);
            }

            throw new Error(getHTTPErrorMessage(response.status));
        }

        if (data?.success === false) {
            throw new Error(extractErrorMessage(data));
        }

        if (!data || typeof data !== "object" || Array.isArray(data)) {
            throw new Error("The AI returned an invalid response.");
        }

        return data;
    } finally {
        clearTimeout(timeout);
        analysisAbortController = null;
    }
}

/* ============================================================
   Submit
   ============================================================ */

if (resumeForm) {
    resumeForm.addEventListener("submit", async event => {
        event.preventDefault();

        if (analysisInProgress) return;

        hideError();

        const file = resumeInput?.files?.[0];
        const validation = validateFile(file);

        if (!validation.valid) {
            showError(validation.message);
            return;
        }

        resetResults();
        analysisInProgress = true;
        setLoadingState(true);

        try {
            const formData = new FormData();
            formData.append("file", file);

            console.log("Starting TalentMatch AI analysis...");

            const data = await fetchAnalysis(formData);
            displayResults(data);
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
   Results
   ============================================================ */

function getClientMatchLevel(score) {
    score = normalizeScore(score);

    if (score >= 85) return "Excellent Match";
    if (score >= 70) return "Strong Match";
    if (score >= 55) return "Moderate Match";
    if (score >= 40) return "Developing Match";
    return "Low Match";
}

function calculateAverageScore(jobs) {
    if (!jobs.length) return 0;

    const scores = jobs
        .map(job => job && typeof job === "object"
            ? normalizeScore(job.match_score ?? job.score ?? 0)
            : 0)
        .filter(Number.isFinite);

    if (!scores.length) return 0;

    return Number(
        (scores.reduce((sum, score) => sum + score, 0) / scores.length).toFixed(2)
    );
}

function displayResults(data) {
    if (!data || typeof data !== "object") {
        showError("The AI returned an empty or invalid response.");
        return;
    }

    const summary = data.summary && typeof data.summary === "object"
        ? data.summary
        : {};

    const jobs = normalizeArray(data.jobs);
    const interviews = normalizeArray(
        data.interviews || data.interview_questions
    );

    let analyzedCount = valueToNumber(summary.jobs_analyzed);
    if (analyzedCount <= 0) analyzedCount = jobs.length;

    if (jobsAnalyzed) {
        jobsAnalyzed.textContent = String(Math.round(analyzedCount));
    }

    let bestScoreValue =
        summary.best_match_score ??
        summary.best_score;

    if ((bestScoreValue === null || bestScoreValue === undefined) && jobs.length) {
        const job = jobs[0];

        bestScoreValue = job && typeof job === "object"
            ? job.match_score ?? job.score ?? 0
            : 0;
    }

    const best = normalizeScore(bestScoreValue);

    if (bestMatchScore) bestMatchScore.textContent = `${best}%`;

    if (bestMatchLevel) {
        bestMatchLevel.textContent = valueToText(
            summary.best_match_level || getClientMatchLevel(best)
        );
    }

    let averageValue =
        summary.average_match_score ??
        summary.average_score;

    if (averageValue === null || averageValue === undefined) {
        averageValue = calculateAverageScore(jobs);
    }

    const average = normalizeScore(averageValue);

    if (averageMatchScore) {
        averageMatchScore.textContent = `${average}%`;
    }

    renderJobs(jobs);
    renderInterviews(interviews);

    if (resultsSection) {
        resultsSection.classList.remove("hidden");

        setTimeout(() => {
            resultsSection.scrollIntoView({
                behavior: "smooth",
                block: "start"
            });
        }, 100);
    }
}

/* ============================================================
   Jobs
   ============================================================ */

function renderJobs(jobs) {
    if (!jobsContainer) return;

    jobsContainer.innerHTML = "";

    if (!jobs.length) {
        jobsContainer.innerHTML = `
            <div class="empty-state">
                <h3>No matching jobs found</h3>
                <p>The system could not identify suitable opportunities for this resume.</p>
            </div>`;
        return;
    }

    jobs.forEach((job, index) => {
        jobsContainer.appendChild(createJobCard(job, index));
    });
}

function createJobCard(job, index) {
    if (!job || typeof job !== "object") {
        job = { title: valueToText(job) };
    }

    const card = document.createElement("article");
    card.className = "job-card";

    const score = normalizeScore(job.match_score ?? job.score ?? 0);
    const semantic = normalizeScore(
        job.semantic_score ??
        job.semantic_match ??
        job.semantic_similarity ??
        0
    );
    const keyword = normalizeScore(
        job.keyword_score ??
        job.tfidf_score ??
        job.lexical_score ??
        job.keyword_match ??
        0
    );

    const rank = valueToText(job.rank || index + 1);

    const category = escapeHTML(
        job.category ||
        job.title ||
        job.role ||
        "Job Opportunity"
    );

    const description = escapeHTML(
        job.description || "No job description available."
    );

    const requirements = escapeHTML(
        job.requirements ||
        job.required_skills ||
        "No requirements provided."
    );

    const benefits = escapeHTML(job.benefits || "");

    const matchLevel = escapeHTML(
        job.match_level || getClientMatchLevel(score)
    );

    const strengths = normalizeArray(
        job.strengths ||
        job.matching_strengths ||
        []
    );

    const strengthsHTML = strengths.length ? `
        <div class="strengths">
            <strong>Why this match:</strong>
            <ul>
                ${strengths.map(strength => `
                    <li>${escapeHTML(strength)}</li>
                `).join("")}
            </ul>
        </div>` : "";

    const matchedSkills = normalizeArray(
        job.matched_skills ||
        job.matching_skills ||
        job.matched ||
        []
    );

    const missingSkills = normalizeArray(
        job.missing_skills ||
        job.skills_to_develop ||
        job.missing ||
        []
    );

    const skillsHTML =
        matchedSkills.length || missingSkills.length ? `
        <div class="skill-analysis">
            ${matchedSkills.length ? `
                <div class="skill-group">
                    <strong>Matching Skills</strong>
                    <p>${escapeHTML(matchedSkills)}</p>
                </div>` : ""}

            ${missingSkills.length ? `
                <div class="skill-group">
                    <strong>Skills to Develop</strong>
                    <p>${escapeHTML(missingSkills)}</p>
                </div>` : ""}
        </div>` : "";

    const skillDetails = normalizeArray(job.skill_details || []);

    const skillDetailsHTML = skillDetails.length ? `
        <div class="skill-details">
            <strong>Skill Evidence</strong>
            <ul>
                ${skillDetails.map(detail => `
                    <li>${escapeHTML(detail)}</li>
                `).join("")}
            </ul>
        </div>` : "";

    card.innerHTML = `
        <div class="job-card-top">
            <div class="job-rank">#${escapeHTML(rank)}</div>

            <div class="job-title-area">
                <h4>${category}</h4>
                <span class="match-level">${matchLevel}</span>
            </div>

            <div class="match-score">
                <strong>${score}%</strong>
                <span>Match</span>
            </div>
        </div>

        <div class="score-bar">
            <div class="score-fill" style="width:${score}%"></div>
        </div>

        <div class="job-signals">
            <div class="signal">
                <span>Semantic</span>
                <strong>${semantic}%</strong>
            </div>

            <div class="signal">
                <span>Keywords</span>
                <strong>${keyword}%</strong>
            </div>
        </div>

        <div class="job-content">
            <div class="job-detail">
                <h5>Job Description</h5>
                <p>${description}</p>
            </div>

            <div class="job-detail">
                <h5>Requirements</h5>
                <p>${requirements}</p>
            </div>

            ${benefits ? `
                <div class="job-detail">
                    <h5>Benefits</h5>
                    <p>${benefits}</p>
                </div>` : ""}

            ${strengthsHTML}
            ${skillsHTML}
            ${skillDetailsHTML}
        </div>`;

    return card;
}

/* ============================================================
   Interviews
   ============================================================ */

function renderInterviews(questions) {
    if (!interviewContainer) return;

    interviewContainer.innerHTML = "";

    if (!questions.length) {
        if (interviewSection) interviewSection.classList.add("hidden");
        return;
    }

    if (interviewSection) {
        interviewSection.classList.remove("hidden");
    }

    questions.forEach((question, index) => {
        interviewContainer.appendChild(
            createInterviewCard(question, index)
        );
    });
}

function createInterviewCard(question, index) {
    if (!question || typeof question !== "object") {
        question = { question: valueToText(question) };
    }

    const card = document.createElement("article");
    card.className = "interview-card";

    const rank = valueToText(question.rank || index + 1);

    const relevance = normalizeScore(
        question.relevance_score ??
        question.score ??
        question.similarity ??
        0
    );

    const questionText = escapeHTML(
        question.question ||
        question.text ||
        question.prompt ||
        ""
    );

    const answer = escapeHTML(
        question.ideal_answer ||
        question.answer ||
        question.guidance ||
        ""
    );

    const role = escapeHTML(question.role || "");
    const category = escapeHTML(question.category || "");
    const difficulty = escapeHTML(question.difficulty || "");
    const experience = escapeHTML(question.experience || "");

    card.innerHTML = `
        <div class="interview-header">
            <div class="interview-number">${escapeHTML(rank)}</div>

            <div class="interview-meta">
                ${role ? `<span>${role}</span>` : ""}
                ${category ? `<span>${category}</span>` : ""}
                ${difficulty ? `<span>${difficulty}</span>` : ""}
                ${experience ? `<span>${experience}</span>` : ""}
            </div>

            <div class="interview-score">${relevance}%</div>
        </div>

        <div class="interview-question">
            <strong>Interview Question</strong>
            <p>${questionText}</p>
        </div>

        ${answer ? `
            <details class="ideal-answer">
                <summary>View ideal answer guidance</summary>
                <p>${answer}</p>
            </details>` : ""}`;

    return card;
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

        window.scrollTo({
            top: 0,
            behavior: "smooth"
        });
    });
}

/* ============================================================
   Startup
   ============================================================ */

document.addEventListener("DOMContentLoaded", () => {
    resetResults();

    console.log("TalentMatch AI frontend initialized.");
    console.log("Frontend: Vercel");
    console.log("AI backend:", API_BASE_URL);
    console.log("Analysis timeout:", ANALYSIS_TIMEOUT_MS / 1000, "seconds");

    /*
       Wake Render immediately after the Vercel page loads.
       This runs in the background and does not delay the UI.
    */
    wakeBackend();
});
