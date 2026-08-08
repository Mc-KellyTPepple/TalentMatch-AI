/*
===============================================================
TalentMatch AI
Production Frontend Application

Responsibilities:
• Resume upload
• File validation
• API communication
• Loading state
• Display AI job matches
• Display candidate summary
• Display extracted skills
• Display interview recommendations
• Safe rendering of backend objects
• Robust error handling
• Prevent [object Object] display
===============================================================
*/

"use strict";

// =============================================================
// Configuration
// =============================================================

const API_ENDPOINT = "/analyze";

const MAX_FILE_SIZE = 10 * 1024 * 1024;

const SUPPORTED_EXTENSIONS = [
    ".pdf",
    ".docx",
    ".txt"
];

// =============================================================
// DOM Elements
// =============================================================

const resumeForm =
    document.getElementById("resumeForm");

const resumeInput =
    document.getElementById("resume");

const uploadArea =
    document.getElementById("uploadArea");

const selectedFile =
    document.getElementById("selectedFile");

const fileName =
    document.getElementById("fileName");

const fileSize =
    document.getElementById("fileSize");

const removeFile =
    document.getElementById("removeFile");

const analyzeButton =
    document.getElementById("analyzeButton");

const buttonText =
    document.getElementById("buttonText");

const buttonLoader =
    document.getElementById("buttonLoader");

const errorMessage =
    document.getElementById("errorMessage");

const loadingSection =
    document.getElementById("loadingSection");

const resultsSection =
    document.getElementById("resultsSection");

const newAnalysis =
    document.getElementById("newAnalysis");

const jobsContainer =
    document.getElementById("jobsContainer");

const interviewSection =
    document.getElementById("interviewSection");

const interviewContainer =
    document.getElementById("interviewContainer");

const jobsAnalyzed =
    document.getElementById("jobsAnalyzed");

const bestMatchScore =
    document.getElementById("bestMatchScore");

const bestMatchLevel =
    document.getElementById("bestMatchLevel");

const averageMatchScore =
    document.getElementById("averageMatchScore");

// =============================================================
// Safe Value Conversion
// =============================================================

/*
IMPORTANT:

The backend may return:

    string
    number
    boolean
    list
    dictionary/object
    null

JavaScript normally converts an object to:

    [object Object]

This function NEVER does that.

Objects are converted recursively into readable text.
*/

function valueToText(value) {

    if (
        value === null ||
        value === undefined
    ) {
        return "";
    }

    if (typeof value === "string") {
        return value;
    }

    if (
        typeof value === "number" ||
        typeof value === "boolean"
    ) {
        return String(value);
    }

    if (Array.isArray(value)) {

        return value
            .map(item => valueToText(item))
            .filter(text => text.length > 0)
            .join(", ");
    }

    if (typeof value === "object") {

        const entries =
            Object.entries(value);

        return entries
            .map(([key, item]) => {

                const text =
                    valueToText(item);

                if (!text) {
                    return "";
                }

                return `${key}: ${text}`;

            })
            .filter(text => text.length > 0)
            .join("; ");
    }

    return String(value);
}

// =============================================================
// Safe Number Conversion
// =============================================================

function valueToNumber(value) {

    if (
        value === null ||
        value === undefined
    ) {
        return 0;
    }

    if (
        typeof value === "object"
    ) {
        return 0;
    }

    const number =
        Number(value);

    if (
        !Number.isFinite(number)
    ) {
        return 0;
    }

    return number;
}

// =============================================================
// HTML Escaping
// =============================================================

function escapeHTML(value) {

    const text =
        valueToText(value);

    const div =
        document.createElement("div");

    div.textContent =
        text;

    return div.innerHTML;
}

// =============================================================
// Array Normalization
// =============================================================

function normalizeArray(value) {

    if (Array.isArray(value)) {
        return value;
    }

    if (
        value === null ||
        value === undefined ||
        value === ""
    ) {
        return [];
    }

    return [value];
}

// =============================================================
// Score Normalization
// =============================================================

function normalizeScore(value) {

    let score =
        valueToNumber(value);

    /*
    Backend may return:

        0.87
        87
        "87"
        "0.87"
    */

    if (
        score > 0 &&
        score <= 1
    ) {
        score *= 100;
    }

    score =
        Math.max(
            0,
            Math.min(
                100,
                score
            )
        );

    return Number(
        score.toFixed(2)
    );
}

// =============================================================
// File Size Formatting
// =============================================================

function formatFileSize(bytes) {

    if (bytes < 1024) {
        return `${bytes} B`;
    }

    if (
        bytes <
        1024 * 1024
    ) {
        return `${(
            bytes / 1024
        ).toFixed(1)} KB`;
    }

    return `${(
        bytes /
        (1024 * 1024)
    ).toFixed(2)} MB`;
}

// =============================================================
// File Validation
// =============================================================

function validateFile(file) {

    if (!file) {

        return {
            valid: false,
            message:
                "Please select a resume."
        };
    }

    const extension =
        "." +
        file.name
            .split(".")
            .pop()
            .toLowerCase();

    if (
        !SUPPORTED_EXTENSIONS.includes(
            extension
        )
    ) {

        return {
            valid: false,
            message:
                "Unsupported file format. " +
                "Please upload a PDF, DOCX or TXT resume."
        };
    }

    if (
        file.size <= 0
    ) {

        return {
            valid: false,
            message:
                "The selected file is empty."
        };
    }

    if (
        file.size >
        MAX_FILE_SIZE
    ) {

        return {
            valid: false,
            message:
                "The resume is larger than the 10 MB limit."
        };
    }

    return {
        valid: true,
        message: ""
    };
}

// =============================================================
// Error Handling
// =============================================================

function showError(message) {

    if (!errorMessage) {
        return;
    }

    const text =
        valueToText(message);

    errorMessage.textContent =
        text ||
        "An unexpected error occurred.";

    errorMessage.classList.remove(
        "hidden"
    );
}

function hideError() {

    if (!errorMessage) {
        return;
    }

    errorMessage.textContent =
        "";

    errorMessage.classList.add(
        "hidden"
    );
}

// =============================================================
// Selected File
// =============================================================

function displaySelectedFile(file) {

    if (!file) {
        return;
    }

    fileName.textContent =
        valueToText(
            file.name
        );

    fileSize.textContent =
        formatFileSize(
            file.size
        );

    selectedFile.classList.remove(
        "hidden"
    );
}

function clearSelectedFile() {

    resumeInput.value =
        "";

    selectedFile.classList.add(
        "hidden"
    );

    fileName.textContent =
        "";

    fileSize.textContent =
        "";
}

// =============================================================
// File Selection
// =============================================================

resumeInput.addEventListener(
    "change",
    function () {

        hideError();

        const file =
            resumeInput.files[0];

        if (!file) {
            return;
        }

        const validation =
            validateFile(file);

        if (!validation.valid) {

            showError(
                validation.message
            );

            clearSelectedFile();

            return;
        }

        displaySelectedFile(file);
    }
);

// =============================================================
// Remove File
// =============================================================

removeFile.addEventListener(
    "click",
    function (event) {

        event.preventDefault();

        clearSelectedFile();

        hideError();
    }
);

// =============================================================
// Drag & Drop
// =============================================================

uploadArea.addEventListener(
    "dragover",
    function (event) {

        event.preventDefault();

        uploadArea.classList.add(
            "dragging"
        );
    }
);

uploadArea.addEventListener(
    "dragleave",
    function () {

        uploadArea.classList.remove(
            "dragging"
        );
    }
);

uploadArea.addEventListener(
    "drop",
    function (event) {

        event.preventDefault();

        uploadArea.classList.remove(
            "dragging"
        );

        hideError();

        const files =
            event.dataTransfer.files;

        if (
            !files ||
            !files.length
        ) {
            return;
        }

        const file =
            files[0];

        const validation =
            validateFile(file);

        if (!validation.valid) {

            showError(
                validation.message
            );

            return;
        }

        try {

            const dataTransfer =
                new DataTransfer();

            dataTransfer.items.add(
                file
            );

            resumeInput.files =
                dataTransfer.files;

        } catch (error) {

            console.warn(
                "Unable to assign dropped file.",
                error
            );
        }

        displaySelectedFile(file);
    }
);

// =============================================================
// Loading State
// =============================================================

function setLoadingState(
    isLoading
) {

    if (isLoading) {

        analyzeButton.disabled =
            true;

        buttonText.textContent =
            "Analyzing...";

        buttonLoader.classList.remove(
            "hidden"
        );

        loadingSection.classList.remove(
            "hidden"
        );

    } else {

        analyzeButton.disabled =
            false;

        buttonText.textContent =
            "Analyze Resume";

        buttonLoader.classList.add(
            "hidden"
        );

        loadingSection.classList.add(
            "hidden"
        );
    }
}

// =============================================================
// Reset Results
// =============================================================

function resetResults() {

    jobsContainer.innerHTML =
        "";

    interviewContainer.innerHTML =
        "";

    jobsAnalyzed.textContent =
        "0";

    bestMatchScore.textContent =
        "0%";

    bestMatchLevel.textContent =
        "No Match";

    averageMatchScore.textContent =
        "0%";

    interviewSection.classList.add(
        "hidden"
    );

    resultsSection.classList.add(
        "hidden"
    );
}

// =============================================================
// Extract Error Message From Backend
// =============================================================

function extractErrorMessage(data) {

    if (!data) {

        return (
            "The server returned an empty response."
        );
    }

    /*
    FastAPI commonly returns:

        {
            "detail": "..."
        }

    Your application may return:

        {
            "error": "..."
        }

    It may also return an object.
    */

    const candidates = [
        data.error,
        data.detail,
        data.message
    ];

    for (
        const candidate
        of candidates
    ) {

        const text =
            valueToText(
                candidate
            );

        if (text) {
            return text;
        }
    }

    return (
        "Resume analysis failed."
    );
}

// =============================================================
// Submit Resume
// =============================================================

resumeForm.addEventListener(
    "submit",
    async function (event) {

        event.preventDefault();

        hideError();

        const file =
            resumeInput.files[0];

        const validation =
            validateFile(file);

        if (!validation.valid) {

            showError(
                validation.message
            );

            return;
        }

        resetResults();

        setLoadingState(true);

        try {

            /*
            IMPORTANT:

            FastAPI endpoint:

                file: UploadFile = File(...)

            Therefore the multipart
            field MUST be:

                "file"
            */

            const formData =
                new FormData();

            formData.append(
                "file",
                file
            );

            const response =
                await fetch(
                    API_ENDPOINT,
                    {
                        method: "POST",
                        body: formData
                    }
                );

            let data;

            /*
            Always attempt JSON first.
            */

            const responseText =
                await response.text();

            try {

                data =
                    responseText
                        ? JSON.parse(
                            responseText
                        )
                        : null;

            } catch (
                jsonError
            ) {

                console.error(
                    "Invalid JSON response:",
                    responseText
                );

                throw new Error(
                    "The server returned an invalid response."
                );
            }

            console.log(
                "TalentMatch AI response:",
                data
            );

            /*
            HTTP-level error
            */

            if (!response.ok) {

                throw new Error(
                    extractErrorMessage(
                        data
                    )
                );
            }

            /*
            Application-level error
            */

            if (
                data &&
                data.success === false
            ) {

                throw new Error(
                    extractErrorMessage(
                        data
                    )
                );
            }

            /*
            Validate expected response.
            */

            if (
                !data ||
                typeof data !== "object"
            ) {

                throw new Error(
                    "The AI returned an invalid response."
                );
            }

            displayResults(data);

        } catch (error) {

            console.error(
                "TalentMatch AI error:",
                error
            );

            showError(
                error instanceof Error
                    ? error.message
                    : valueToText(error)
            );

        } finally {

            setLoadingState(false);
        }
    }
);

// =============================================================
// Display Results
// =============================================================

function displayResults(data) {

    if (
        !data ||
        typeof data !== "object"
    ) {

        showError(
            "The AI returned an empty or invalid response."
        );

        return;
    }

    console.log(
        "Displaying TalentMatch AI results:",
        data
    );

    const summary =
        (
            data.summary &&
            typeof data.summary === "object"
        )
            ? data.summary
            : {};

    const jobs =
        normalizeArray(
            data.jobs
        );

    const interviews =
        normalizeArray(
            data.interviews ||
            data.interview_questions
        );

    // =========================================================
    // Jobs Analyzed
    // =========================================================

    let analyzedCount =
        valueToNumber(
            summary.jobs_analyzed
        );

    /*
    If the backend does not provide the number,
    use the number of returned jobs.
    */

    if (
        analyzedCount <= 0
    ) {
        analyzedCount =
            jobs.length;
    }

    jobsAnalyzed.textContent =
        String(
            Math.round(
                analyzedCount
            )
        );

    // =========================================================
    // Best Match
    // =========================================================

    let bestScoreValue =
        summary.best_match_score;

    if (
        bestScoreValue === null ||
        bestScoreValue === undefined
    ) {

        bestScoreValue =
            summary.best_score;
    }

    if (
        (
            bestScoreValue === null ||
            bestScoreValue === undefined
        ) &&
        jobs.length
    ) {

        bestScoreValue =
            jobs[0] &&
            typeof jobs[0] === "object"
                ? (
                    jobs[0].match_score ??
                    jobs[0].score ??
                    0
                )
                : 0;
    }

    const best =
        normalizeScore(
            bestScoreValue
        );

    bestMatchScore.textContent =
        `${best}%`;

    const backendMatchLevel =
        summary.best_match_level;

    bestMatchLevel.textContent =
        escapeHTML(
            backendMatchLevel ||
            getClientMatchLevel(best)
        );

    // =========================================================
    // Average Match
    // =========================================================

    let averageValue =
        summary.average_match_score;

    if (
        averageValue === null ||
        averageValue === undefined
    ) {

        averageValue =
            summary.average_score;
    }

    if (
        averageValue === null ||
        averageValue === undefined
    ) {

        averageValue =
            calculateAverageScore(
                jobs
            );
    }

    const average =
        normalizeScore(
            averageValue
        );

    averageMatchScore.textContent =
        `${average}%`;

    // =========================================================
    // Render
    // =========================================================

    renderJobs(
        jobs
    );

    renderInterviews(
        interviews
    );

    resultsSection.classList.remove(
        "hidden"
    );

    // =========================================================
    // Scroll to Results
    // =========================================================

    setTimeout(
        function () {

            resultsSection.scrollIntoView({
                behavior: "smooth",
                block: "start"
            });

        },
        100
    );
}

// =============================================================
// Calculate Average Score
// =============================================================

function calculateAverageScore(
    jobs
) {

    if (
        !jobs.length
    ) {
        return 0;
    }

    const scores =
        jobs.map(
            function (job) {

                if (
                    !job ||
                    typeof job !== "object"
                ) {
                    return 0;
                }

                return normalizeScore(
                    job.match_score ??
                    job.score ??
                    0
                );
            }
        );

    const validScores =
        scores.filter(
            score =>
                Number.isFinite(
                    score
                )
        );

    if (
        !validScores.length
    ) {
        return 0;
    }

    const total =
        validScores.reduce(
            (
                sum,
                score
            ) =>
                sum + score,
            0
        );

    return Number(
        (
            total /
            validScores.length
        ).toFixed(2)
    );
}

// =============================================================
// Render Jobs
// =============================================================

function renderJobs(
    jobs
) {

    jobsContainer.innerHTML =
        "";

    if (
        !jobs.length
    ) {

        jobsContainer.innerHTML = `
            <div class="empty-state">

                <h3>
                    No matching jobs found
                </h3>

                <p>
                    The system could not identify
                    suitable opportunities for this resume.
                </p>

            </div>
        `;

        return;
    }

    jobs.forEach(
        function (
            job,
            index
        ) {

            const card =
                createJobCard(
                    job,
                    index
                );

            jobsContainer.appendChild(
                card
            );
        }
    );
}

// =============================================================
// Create Job Card
// =============================================================

function createJobCard(
    job,
    index
) {

    /*
    Guarantee that job is an object.
    */

    if (
        !job ||
        typeof job !== "object"
    ) {

        job = {
            title:
                valueToText(
                    job
                )
        };
    }

    const card =
        document.createElement(
            "article"
        );

    card.className =
        "job-card";

    // =========================================================
    // Scores
    // =========================================================

    const score =
        normalizeScore(
            job.match_score ??
            job.score ??
            0
        );

    const semantic =
        normalizeScore(
            job.semantic_score ??
            job.semantic_match ??
            job.semantic_similarity ??
            0
        );

    const keyword =
        normalizeScore(
            job.keyword_score ??
            job.tfidf_score ??
            job.lexical_score ??
            job.keyword_match ??
            0
        );

    // =========================================================
    // Basic Information
    // =========================================================

    const rank =
        valueToText(
            job.rank ||
            index + 1
        );

    const category =
        escapeHTML(
            job.category ||
            job.title ||
            job.role ||
            "Job Opportunity"
        );

    const description =
        escapeHTML(
            job.description ||
            "No job description available."
        );

    const requirements =
        escapeHTML(
            job.requirements ||
            job.required_skills ||
            "No requirements provided."
        );

    const benefits =
        escapeHTML(
            job.benefits ||
            ""
        );

    const matchLevel =
        escapeHTML(
            job.match_level ||
            getClientMatchLevel(
                score
            )
        );

    // =========================================================
    // Strengths
    // =========================================================

    const strengths =
        normalizeArray(
            job.strengths ||
            job.matching_strengths ||
            []
        );

    const strengthsHTML =
        strengths.length
            ? `
                <div class="strengths">

                    <strong>
                        Why this match:
                    </strong>

                    <ul>

                        ${strengths
                            .map(
                                function (
                                    strength
                                ) {

                                    return `
                                        <li>
                                            ${escapeHTML(
                                                strength
                                            )}
                                        </li>
                                    `;
                                }
                            )
                            .join("")
                        }

                    </ul>

                </div>
            `
            : "";

    // =========================================================
    // Matching Skills
    // =========================================================

    const matchedSkills =
        normalizeArray(
            job.matched_skills ||
            job.matching_skills ||
            job.matched ||
            []
        );

    const missingSkills =
        normalizeArray(
            job.missing_skills ||
            job.skills_to_develop ||
            job.missing ||
            []
        );

    const skillsHTML =
        (
            matchedSkills.length ||
            missingSkills.length
        )
            ? `
                <div class="skill-analysis">

                    ${
                        matchedSkills.length
                            ? `
                                <div class="skill-group">

                                    <strong>
                                        Matching Skills
                                    </strong>

                                    <p>
                                        ${escapeHTML(
                                            matchedSkills
                                        )}
                                    </p>

                                </div>
                            `
                            : ""
                    }

                    ${
                        missingSkills.length
                            ? `
                                <div class="skill-group">

                                    <strong>
                                        Skills to Develop
                                    </strong>

                                    <p>
                                        ${escapeHTML(
                                            missingSkills
                                        )}
                                    </p>

                                </div>
                            `
                            : ""
                    }

                </div>
            `
            : "";

    // =========================================================
    // Candidate Fit / Skill Details
    // =========================================================

    const skillDetails =
        normalizeArray(
            job.skill_details ||
            []
        );

    const skillDetailsHTML =
        skillDetails.length
            ? `
                <div class="skill-details">

                    <strong>
                        Skill Evidence
                    </strong>

                    <ul>

                        ${skillDetails
                            .map(
                                function (
                                    detail
                                ) {

                                    return `
                                        <li>
                                            ${escapeHTML(
                                                detail
                                            )}
                                        </li>
                                    `;
                                }
                            )
                            .join("")
                        }

                    </ul>

                </div>
            `
            : "";

    // =========================================================
    // Build Card
    // =========================================================

    card.innerHTML = `

        <div class="job-card-top">

            <div class="job-rank">
                #${escapeHTML(rank)}
            </div>

            <div class="job-title-area">

                <h4>
                    ${category}
                </h4>

                <span class="match-level">
                    ${matchLevel}
                </span>

            </div>

            <div class="match-score">

                <strong>
                    ${score}%
                </strong>

                <span>
                    Match
                </span>

            </div>

        </div>


        <div class="score-bar">

            <div
                class="score-fill"
                style="width: ${score}%"
            ></div>

        </div>


        <div class="job-signals">

            <div class="signal">

                <span>
                    Semantic
                </span>

                <strong>
                    ${semantic}%
                </strong>

            </div>


            <div class="signal">

                <span>
                    Keywords
                </span>

                <strong>
                    ${keyword}%
                </strong>

            </div>

        </div>


        <div class="job-content">

            <div class="job-detail">

                <h5>
                    Job Description
                </h5>

                <p>
                    ${description}
                </p>

            </div>


            <div class="job-detail">

                <h5>
                    Requirements
                </h5>

                <p>
                    ${requirements}
                </p>

            </div>


            ${
                benefits
                    ? `
                        <div class="job-detail">

                            <h5>
                                Benefits
                            </h5>

                            <p>
                                ${benefits}
                            </p>

                        </div>
                    `
                    : ""
            }


            ${strengthsHTML}


            ${skillsHTML}


            ${skillDetailsHTML}

        </div>
    `;

    return card;
}

// =============================================================
// Client Match Level
// =============================================================

function getClientMatchLevel(
    score
) {

    score =
        normalizeScore(
            score
        );

    if (
        score >= 85
    ) {
        return "Excellent Match";
    }

    if (
        score >= 70
    ) {
        return "Strong Match";
    }

    if (
        score >= 55
    ) {
        return "Moderate Match";
    }

    if (
        score >= 40
    ) {
        return "Developing Match";
    }

    return "Low Match";
}

// =============================================================
// Render Interviews
// =============================================================

function renderInterviews(
    questions
) {

    interviewContainer.innerHTML =
        "";

    if (
        !questions.length
    ) {

        interviewSection.classList.add(
            "hidden"
        );

        return;
    }

    interviewSection.classList.remove(
        "hidden"
    );

    questions.forEach(
        function (
            question,
            index
        ) {

            const card =
                createInterviewCard(
                    question,
                    index
                );

            interviewContainer.appendChild(
                card
            );
        }
    );
}

// =============================================================
// Create Interview Card
// =============================================================

function createInterviewCard(
    question,
    index
) {

    /*
    Guarantee object safety.
    */

    if (
        !question ||
        typeof question !== "object"
    ) {

        question = {
            question:
                valueToText(
                    question
                )
        };
    }

    const card =
        document.createElement(
            "article"
        );

    card.className =
        "interview-card";

    const rank =
        valueToText(
            question.rank ||
            index + 1
        );

    const relevance =
        normalizeScore(
            question.relevance_score ??
            question.score ??
            question.similarity ??
            0
        );

    const questionText =
        escapeHTML(
            question.question ||
            question.text ||
            question.prompt ||
            ""
        );

    const answer =
        escapeHTML(
            question.ideal_answer ||
            question.answer ||
            question.guidance ||
            ""
        );

    const role =
        escapeHTML(
            question.role ||
            ""
        );

    const category =
        escapeHTML(
            question.category ||
            ""
        );

    const difficulty =
        escapeHTML(
            question.difficulty ||
            ""
        );

    const experience =
        escapeHTML(
            question.experience ||
            ""
        );

    card.innerHTML = `

        <div class="interview-header">

            <div class="interview-number">
                ${escapeHTML(rank)}
            </div>

            <div class="interview-meta">

                ${
                    role
                        ? `
                            <span>
                                ${role}
                            </span>
                        `
                        : ""
                }

                ${
                    category
                        ? `
                            <span>
                                ${category}
                            </span>
                        `
                        : ""
                }

                ${
                    difficulty
                        ? `
                            <span>
                                ${difficulty}
                            </span>
                        `
                        : ""
                }

                ${
                    experience
                        ? `
                            <span>
                                ${experience}
                            </span>
                        `
                        : ""
                }

            </div>

            <div class="interview-score">
                ${relevance}%
            </div>

        </div>


        <div class="interview-question">

            <strong>
                Interview Question
            </strong>

            <p>
                ${questionText}
            </p>

        </div>


        ${
            answer
                ? `
                    <details class="ideal-answer">

                        <summary>
                            View ideal answer guidance
                        </summary>

                        <p>
                            ${answer}
                        </p>

                    </details>
                `
                : ""
        }

    `;

    return card;
}

// =============================================================
// New Analysis
// =============================================================

newAnalysis.addEventListener(
    "click",
    function () {

        resetResults();

        clearSelectedFile();

        hideError();

        window.scrollTo({
            top: 0,
            behavior: "smooth"
        });
    }
);

// =============================================================
// Prevent Browser Navigation During Drag & Drop
// =============================================================

window.addEventListener(
    "dragover",
    function (event) {

        event.preventDefault();
    }
);

window.addEventListener(
    "drop",
    function (event) {

        event.preventDefault();
    }
);

// =============================================================
// Startup
// =============================================================

document.addEventListener(
    "DOMContentLoaded",
    function () {

        resetResults();

        console.log(
            "TalentMatch AI frontend initialized successfully."
        );
    }
);
