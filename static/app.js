/*
=============================================================
TalentMatch AI
Production Frontend Application
=============================================================

Designed for:

- FastAPI backend
- Render Free
- CPU inference
- Lazy model loading
- PDF / DOCX / TXT resumes
- Unreliable/slow first AI request
- JSON and non-JSON backend failures
- Safe frontend rendering

IMPORTANT:
The /analyze endpoint may take significantly longer on the
first request because the AI prediction engine is loaded
lazily.

This frontend therefore:

1. Uses an AbortController timeout.
2. Detects empty server responses.
3. Detects HTML/non-JSON responses.
4. Gives useful messages for 500/502/503/504.
5. Never displays [object Object].
6. Prevents duplicate analysis requests.
=============================================================
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

/*
IMPORTANT:

Render may need considerable time to perform the FIRST
analysis because the AI model is loaded lazily.

Do not use an extremely short timeout.

20 minutes gives the backend enough time for:

- cold start
- model loading
- embedding generation
- job ranking
- interview retrieval
*/

const ANALYSIS_TIMEOUT_MS =
    20 * 60 * 1000;


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
// Runtime State
// =============================================================

let analysisInProgress = false;

let analysisAbortController = null;


// =============================================================
// Utility: Safe Value -> Text
// =============================================================

function valueToText(value) {

    if (
        value === null ||
        value === undefined
    ) {
        return "";
    }

    if (
        typeof value === "string"
    ) {
        return value;
    }

    if (
        typeof value === "number" ||
        typeof value === "boolean"
    ) {
        return String(value);
    }

    if (
        Array.isArray(value)
    ) {

        return value
            .map(
                item =>
                    valueToText(item)
            )
            .filter(
                text =>
                    text.length > 0
            )
            .join(", ");
    }

    if (
        typeof value === "object"
    ) {

        return Object.entries(value)
            .map(
                ([key, item]) => {

                    const text =
                        valueToText(item);

                    if (!text) {
                        return "";
                    }

                    return `${key}: ${text}`;
                }
            )
            .filter(
                text =>
                    text.length > 0
            )
            .join("; ");
    }

    return String(value);
}


// =============================================================
// Utility: Safe Number
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
// Utility: HTML Escape
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
// Utility: Normalize Array
// =============================================================

function normalizeArray(value) {

    if (
        Array.isArray(value)
    ) {
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
// Utility: Normalize Score
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
// Utility: File Size
// =============================================================

function formatFileSize(bytes) {

    if (
        bytes < 1024
    ) {
        return `${bytes} B`;
    }

    if (
        bytes < 1024 * 1024
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

    const filename =
        valueToText(
            file.name
        );

    const extension =
        filename
            .includes(".")
            ? "." +
              filename
                  .split(".")
                  .pop()
                  .toLowerCase()
            : "";

    if (
        !SUPPORTED_EXTENSIONS.includes(
            extension
        )
    ) {

        return {
            valid: false,
            message:
                "Unsupported file format. Please upload a PDF, DOCX or TXT resume."
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
        file.size > MAX_FILE_SIZE
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
// Error Display
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

    if (fileName) {

        fileName.textContent =
            valueToText(
                file.name
            );
    }

    if (fileSize) {

        fileSize.textContent =
            formatFileSize(
                file.size
            );
    }

    if (selectedFile) {

        selectedFile.classList.remove(
            "hidden"
        );
    }
}


function clearSelectedFile() {

    if (resumeInput) {

        resumeInput.value =
            "";
    }

    if (selectedFile) {

        selectedFile.classList.add(
            "hidden"
        );
    }

    if (fileName) {

        fileName.textContent =
            "";
    }

    if (fileSize) {

        fileSize.textContent =
            "";
    }
}


// =============================================================
// File Selection
// =============================================================

if (resumeInput) {

    resumeInput.addEventListener(
        "change",
        function () {

            hideError();

            const file =
                resumeInput.files &&
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

            displaySelectedFile(
                file
            );
        }
    );
}


// =============================================================
// Remove File
// =============================================================

if (removeFile) {

    removeFile.addEventListener(
        "click",
        function (event) {

            event.preventDefault();

            clearSelectedFile();

            hideError();
        }
    );
}


// =============================================================
// Drag & Drop
// =============================================================

if (uploadArea) {

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
                event.dataTransfer &&
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

            /*
            Try to assign the dropped file to
            the hidden file input.
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

            displaySelectedFile(
                file
            );
        }
    );
}


// =============================================================
// Loading State
// =============================================================

function setLoadingState(
    isLoading
) {

    if (isLoading) {

        if (analyzeButton) {

            analyzeButton.disabled =
                true;
        }

        if (buttonText) {

            buttonText.textContent =
                "Analyzing...";
        }

        if (buttonLoader) {

            buttonLoader.classList.remove(
                "hidden"
            );
        }

        if (loadingSection) {

            loadingSection.classList.remove(
                "hidden"
            );
        }

    } else {

        if (analyzeButton) {

            analyzeButton.disabled =
                false;
        }

        if (buttonText) {

            buttonText.textContent =
                "Analyze Resume";
        }

        if (buttonLoader) {

            buttonLoader.classList.add(
                "hidden"
            );
        }

        if (loadingSection) {

            loadingSection.classList.add(
                "hidden"
            );
        }
    }
}


// =============================================================
// Reset Results
// =============================================================

function resetResults() {

    if (jobsContainer) {

        jobsContainer.innerHTML =
            "";
    }

    if (interviewContainer) {

        interviewContainer.innerHTML =
            "";
    }

    if (jobsAnalyzed) {

        jobsAnalyzed.textContent =
            "0";
    }

    if (bestMatchScore) {

        bestMatchScore.textContent =
            "0%";
    }

    if (bestMatchLevel) {

        bestMatchLevel.textContent =
            "No Match";
    }

    if (averageMatchScore) {

        averageMatchScore.textContent =
            "0%";
    }

    if (interviewSection) {

        interviewSection.classList.add(
            "hidden"
        );
    }

    if (resultsSection) {

        resultsSection.classList.add(
            "hidden"
        );
    }
}


// =============================================================
// Backend Error Extraction
// =============================================================

function extractErrorMessage(
    data
) {

    if (!data) {

        return (
            "The server returned an empty response."
        );
    }

    /*
    Standard FastAPI response:

        {
            "detail": "..."
        }

    Application response:

        {
            "error": "..."
        }

    */

    if (
        typeof data === "string"
    ) {

        return data.trim() ||
            "The server returned an empty response.";
    }

    if (
        typeof data !== "object"
    ) {

        return valueToText(
            data
        );
    }

    const candidates = [

        data.error,

        data.detail,

        data.message,

        data.msg,

        data.exception
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
// HTTP Error Explanation
// =============================================================

function getHTTPErrorMessage(
    status
) {

    switch (status) {

        case 400:

            return (
                "The uploaded resume could not be processed. " +
                "Please check that the file is a valid PDF, DOCX or TXT document."
            );

        case 413:

            return (
                "The uploaded resume is too large."
            );

        case 429:

            return (
                "The service is currently busy. " +
                "Please wait a moment and try again."
            );

        case 500:

            return (
                "TalentMatch AI encountered an internal server error while analyzing the resume."
            );

        case 502:

            return (
                "The AI service connection was interrupted. " +
                "Please try the analysis again."
            );

        case 503:

            return (
                "TalentMatch AI is temporarily unavailable. " +
                "The AI model may still be loading or the server may have run out of memory."
            );

        case 504:

            return (
                "The analysis request timed out on the server. " +
                "The AI model may still be initializing. Please try again."
            );

        default:

            return (
                `The server returned HTTP ${status}.`
            );
    }
}


// =============================================================
// Fetch JSON Safely
// =============================================================

async function fetchAnalysis(
    formData
) {

    /*
    AbortController allows the frontend to stop waiting
    forever if the Render service becomes unresponsive.
    */

    analysisAbortController =
        new AbortController();

    const timeout =
        setTimeout(
            function () {

                if (
                    analysisAbortController
                ) {

                    analysisAbortController.abort();
                }

            },
            ANALYSIS_TIMEOUT_MS
        );


    try {

        const response =
            await fetch(
                API_ENDPOINT,
                {
                    method: "POST",

                    body: formData,

                    /*
                    DO NOT manually set Content-Type.

                    Browser must generate the multipart
                    boundary automatically.
                    */

                    signal:
                        analysisAbortController.signal,

                    headers: {

                        "Accept":
                            "application/json"
                    }
                }
            );


        /*
        Read the response as text FIRST.

        This prevents:

            response.json()

        from crashing when Render/FastAPI returns:

        - empty body
        - HTML
        - proxy error
        - plain text
        */

        const responseText =
            await response.text();


        console.log(
            "TalentMatch HTTP status:",
            response.status
        );

        console.log(
            "TalentMatch response length:",
            responseText.length
        );


        /*
        EMPTY RESPONSE
        */

        if (
            !responseText ||
            !responseText.trim()
        ) {

            throw new Error(
                response.ok
                    ? (
                        "The server returned an empty response. " +
                        "The AI analysis may have stopped before producing a result."
                    )
                    : getHTTPErrorMessage(
                        response.status
                    )
            );
        }


        /*
        Attempt JSON parsing.
        */

        let data = null;

        try {

            data =
                JSON.parse(
                    responseText
                );

        } catch (
            jsonError
        ) {

            console.error(
                "Non-JSON server response:",
                responseText.substring(
                    0,
                    1000
                )
            );

            /*
            Render or a reverse proxy can sometimes
            return HTML instead of FastAPI JSON.
            */

            if (
                !response.ok
            ) {

                throw new Error(
                    getHTTPErrorMessage(
                        response.status
                    )
                );
            }

            throw new Error(
                "The server returned an invalid response instead of JSON."
            );
        }


        console.log(
            "TalentMatch AI response:",
            data
        );


        /*
        HTTP ERROR
        */

        if (
            !response.ok
        ) {

            const backendMessage =
                extractErrorMessage(
                    data
                );

            /*
            Prefer useful backend error when available.
            */

            if (
                backendMessage &&
                backendMessage !==
                    "Resume analysis failed."
            ) {

                throw new Error(
                    backendMessage
                );
            }

            throw new Error(
                getHTTPErrorMessage(
                    response.status
                )
            );
        }


        /*
        APPLICATION ERROR
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
        Validate that the backend returned an object.
        */

        if (
            !data ||
            typeof data !== "object" ||
            Array.isArray(data)
        ) {

            throw new Error(
                "The AI returned an invalid response."
            );
        }


        return data;

    } finally {

        clearTimeout(
            timeout
        );

        analysisAbortController =
            null;
    }
}


// =============================================================
// User-Friendly Request Error
// =============================================================

function formatRequestError(
    error
) {

    if (!error) {

        return (
            "An unknown error occurred."
        );
    }


    /*
    Timeout / AbortController
    */

    if (
        error.name ===
        "AbortError"
    ) {

        return (
            "The analysis is taking longer than expected. " +
            "The server may be loading the AI model for the first time. " +
            "Please try again."
        );
    }


    /*
    Browser network failure
    */

    if (
        error instanceof TypeError
    ) {

        return (
            "Unable to connect to the TalentMatch AI server. " +
            "Please check your internet connection and try again."
        );
    }


    return (
        valueToText(
            error.message ||
            error
        ) ||
        "Resume analysis failed."
    );
}


// =============================================================
// Submit Resume
// =============================================================

if (resumeForm) {

    resumeForm.addEventListener(
        "submit",
        async function (event) {

            event.preventDefault();

            /*
            Prevent duplicate requests.

            This is important because two simultaneous
            requests could cause two model-loading attempts.
            */

            if (
                analysisInProgress
            ) {

                return;
            }

            hideError();


            const file =
                resumeInput &&
                resumeInput.files &&
                resumeInput.files[0];


            const validation =
                validateFile(file);


            if (
                !validation.valid
            ) {

                showError(
                    validation.message
                );

                return;
            }


            resetResults();


            analysisInProgress =
                true;


            setLoadingState(
                true
            );


            try {

                /*
                FastAPI expects:

                    file: UploadFile = File(...)

                Therefore:

                    formData.append("file", file)
                */

                const formData =
                    new FormData();


                formData.append(
                    "file",
                    file
                );


                console.log(
                    "Starting TalentMatch AI analysis..."
                );


                const data =
                    await fetchAnalysis(
                        formData
                    );


                displayResults(
                    data
                );


            } catch (error) {

                console.error(
                    "TalentMatch AI analysis error:",
                    error
                );


                const message =
                    formatRequestError(
                        error
                    );


                showError(
                    message
                );


            } finally {

                analysisInProgress =
                    false;


                setLoadingState(
                    false
                );
            }
        }
    );
}


// =============================================================
// Display Results
// =============================================================

function displayResults(
    data
) {

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


    if (
        analyzedCount <= 0
    ) {

        analyzedCount =
            jobs.length;
    }


    if (jobsAnalyzed) {

        jobsAnalyzed.textContent =
            String(
                Math.round(
                    analyzedCount
                )
            );
    }


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


    if (bestMatchScore) {

        bestMatchScore.textContent =
            `${best}%`;
    }


    const backendMatchLevel =
        summary.best_match_level;


    if (bestMatchLevel) {

        bestMatchLevel.textContent =
            valueToText(
                backendMatchLevel ||
                getClientMatchLevel(
                    best
                )
            );
    }


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


    if (averageMatchScore) {

        averageMatchScore.textContent =
            `${average}%`;
    }


    // =========================================================
    // Render
    // =========================================================

    renderJobs(
        jobs
    );


    renderInterviews(
        interviews
    );


    if (resultsSection) {

        resultsSection.classList.remove(
            "hidden"
        );
    }


    // =========================================================
    // Scroll to Results
    // =========================================================

    setTimeout(
        function () {

            if (
                resultsSection
            ) {

                resultsSection.scrollIntoView({
                    behavior:
                        "smooth",

                    block:
                        "start"
                });
            }

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

    if (!jobsContainer) {
        return;
    }


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
    // Skill Evidence
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

    if (!interviewContainer) {
        return;
    }


    interviewContainer.innerHTML =
        "";


    if (
        !questions.length
    ) {

        if (interviewSection) {

            interviewSection.classList.add(
                "hidden"
            );
        }

        return;
    }


    if (interviewSection) {

        interviewSection.classList.remove(
            "hidden"
        );
    }


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

if (newAnalysis) {

    newAnalysis.addEventListener(
        "click",
        function () {

            /*
            Do not allow reset while a request is
            currently being processed.
            */

            if (
                analysisInProgress
            ) {

                return;
            }


            resetResults();

            clearSelectedFile();

            hideError();


            window.scrollTo({
                top: 0,
                behavior:
                    "smooth"
            });
        }
    );
}


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

        console.log(
            "Analysis timeout:",
            ANALYSIS_TIMEOUT_MS / 1000,
            "seconds"
        );
    }
);
