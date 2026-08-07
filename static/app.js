/*
===============================================================
TalentMatch AI

Frontend Application

Responsibilities:

• Resume upload
• File validation
• API communication
• Loading state
• Display AI job matches
• Display candidate summary
• Display interview recommendations
• Error handling

Designed for:
Render Free
512 MB RAM

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
// Utility Functions
// =============================================================

function escapeHTML(value) {

    if (value === null || value === undefined) {
        return "";
    }

    const div = document.createElement("div");

    div.textContent = String(value);

    return div.innerHTML;
}


// =============================================================
// File Size Formatting
// =============================================================

function formatFileSize(bytes) {

    if (bytes < 1024) {
        return `${bytes} B`;
    }

    if (bytes < 1024 * 1024) {
        return `${(bytes / 1024).toFixed(1)} KB`;
    }

    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}


// =============================================================
// File Validation
// =============================================================

function validateFile(file) {

    if (!file) {

        return {
            valid: false,
            message: "Please select a resume."
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

    if (file.size <= 0) {

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

    errorMessage.textContent =
        message;

    errorMessage.classList.remove(
        "hidden"
    );

}


function hideError() {

    errorMessage.textContent = "";

    errorMessage.classList.add(
        "hidden"
    );

}


// =============================================================
// Selected File Display
// =============================================================

function displaySelectedFile(file) {

    fileName.textContent =
        file.name;

    fileSize.textContent =
        formatFileSize(file.size);

    selectedFile.classList.remove(
        "hidden"
    );

}


function clearSelectedFile() {

    resumeInput.value = "";

    selectedFile.classList.add(
        "hidden"
    );

    fileName.textContent = "";

    fileSize.textContent = "";

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
// Remove Selected File
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

        if (!files || !files.length) {
            return;
        }

        const file = files[0];

        const validation =
            validateFile(file);

        if (!validation.valid) {

            showError(
                validation.message
            );

            return;
        }

        /*
        DataTransfer is used so that the
        selected file is also available
        to the form submission.
        */

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

function setLoadingState(isLoading) {

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

    jobsContainer.innerHTML = "";

    interviewContainer.innerHTML = "";

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

            const formData =
                new FormData();

            formData.append(
                "resume",
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


            /*
            Attempt to parse JSON regardless
            of HTTP status so the backend can
            return a useful error message.
            */

            let data;

            try {

                data =
                    await response.json();

            } catch (jsonError) {

                throw new Error(
                    "The server returned an invalid response."
                );

            }


            if (!response.ok) {

                const message =
                    data.detail ||
                    data.message ||
                    data.error ||
                    "Resume analysis failed.";

                throw new Error(
                    message
                );

            }


            displayResults(data);

        } catch (error) {

            console.error(
                "TalentMatch AI error:",
                error
            );

            showError(
                error.message ||
                "Unable to analyze the resume. Please try again."
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

    if (!data) {

        showError(
            "The AI returned an empty response."
        );

        return;
    }


    /*
    Expected backend structure:

    {
        summary: {
            jobs_analyzed: 10,
            best_match_score: 87,
            best_match_level: "Excellent Match",
            average_match_score: 74
        },

        jobs: [...],

        interviews: [...]
    }
    */

    const summary =
        data.summary || {};


    jobsAnalyzed.textContent =
        summary.jobs_analyzed ?? 0;


    bestMatchScore.textContent =
        `${summary.best_match_score ?? 0}%`;


    bestMatchLevel.textContent =
        summary.best_match_level ||
        "No Match";


    averageMatchScore.textContent =
        `${summary.average_match_score ?? 0}%`;


    renderJobs(
        data.jobs || []
    );


    renderInterviews(
        data.interviews ||
        data.interview_questions ||
        []
    );


    resultsSection.classList.remove(
        "hidden"
    );


    /*
    Move the user to the results
    without forcing an aggressive jump.
    */

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
// Render Job Matches
// =============================================================

function renderJobs(jobs) {

    jobsContainer.innerHTML = "";

    if (!jobs.length) {

        jobsContainer.innerHTML = `
            <div class="empty-state">
                <h3>No matching jobs found</h3>
                <p>
                    The system could not identify
                    suitable opportunities for this resume.
                </p>
            </div>
        `;

        return;
    }


    jobs.forEach(
        function (job, index) {

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

    const card =
        document.createElement(
            "article"
        );

    card.className =
        "job-card";


    const score =
        Number(
            job.match_score ?? 0
        );


    const semantic =
        Number(
            job.semantic_score ?? 0
        );


    const keyword =
        Number(
            job.keyword_score ?? 0
        );


    const rank =
        job.rank ||
        index + 1;


    const category =
        escapeHTML(
            job.category ||
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
            getClientMatchLevel(score)
        );


    const strengths =
        Array.isArray(
            job.strengths
        )
            ? job.strengths
            : [];


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
                                function (strength) {

                                    return `
                                        <li>
                                            ${escapeHTML(strength)}
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


    card.innerHTML = `

        <div class="job-card-top">

            <div class="job-rank">
                #${rank}
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
                style="width: ${Math.min(
                    100,
                    Math.max(
                        0,
                        score
                    )
                )}%"
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

        </div>

    `;


    return card;

}


// =============================================================
// Client-side Match Level
// =============================================================

function getClientMatchLevel(
    score
) {

    if (score >= 85) {
        return "Excellent Match";
    }

    if (score >= 70) {
        return "Strong Match";
    }

    if (score >= 55) {
        return "Moderate Match";
    }

    if (score >= 40) {
        return "Developing Match";
    }

    return "Low Match";

}


// =============================================================
// Render Interview Questions
// =============================================================

function renderInterviews(
    questions
) {

    interviewContainer.innerHTML = "";

    if (!questions.length) {

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

    const card =
        document.createElement(
            "article"
        );

    card.className =
        "interview-card";


    const rank =
        question.rank ||
        index + 1;


    const relevance =
        Number(
            question.relevance_score ??
            question.score ??
            0
        );


    const questionText =
        escapeHTML(
            question.question ||
            ""
        );


    const answer =
        escapeHTML(
            question.ideal_answer ||
            question.answer ||
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


    card.innerHTML = `

        <div class="interview-header">

            <div class="interview-number">
                ${rank}
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
// Prevent accidental browser navigation
// during drag & drop
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
            "TalentMatch AI frontend initialized."
        );

    }
);
