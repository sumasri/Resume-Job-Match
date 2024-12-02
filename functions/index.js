const functions = require("firebase-functions");
const express = require("express");
const cors = require("cors");
const { OpenAI } = require("openai");
const pdfParse = require("pdf-parse");
const admin = require("firebase-admin");
const Busboy = require("busboy");
const { spawn } = require("child_process");
const path = require("path");

admin.initializeApp();

const runtimeOpts = {
  timeoutSeconds: 300,
  memory: "1GB",
};

const app = express();

app.use(
  cors({
    methods: ["GET", "POST", "OPTIONS"],
    allowedHeaders: ["Content-Type"],
  })
);

app.use(express.json());

const openai = new OpenAI({
  apiKey: functions.config().openai?.key || process.env.OPENAI_API_KEY,
});

app.post("/parse", async (req, res) => {
  console.log("[Parse] Starting resume parsing process");
  try {
    const { text, filename } = req.body;

    if (!text) {
      console.log("[Parse] Error: No text provided");
      return res.status(400).json({ error: "No text provided" });
    }

    console.log("[Parse] Processing resume:", {
      filename,
      textLength: text.length,
    });

    const systemPrompt = `You are a resume parsing assistant. Extract information from the resume text and format it EXACTLY according to this JSON schema:
      {
          "pii": {
            "full_name": "string",
            "email": "string",
            "phone": "string"
          },
          "education": [
            {
              "organization": "string",
              "degree": "string",
              "major": "string",
              "start_date": "YYYY-MM-DD or null",
              "end_date": "YYYY-MM-DD or null",
              "achievements": ["string"]
            }
          ],
          "work_experience": [
            {
              "job_title": "string",
              "company_name": "string",
              "location": "string",
              "start_date": "YYYY-MM-DD or null",
              "end_date": "YYYY-MM-DD or 'present'",
              "bullet_points": ["string"]
            }
          ],
          "skills": {
            "Programming Languages": ["string"],
            "Tools": ["string"],
            "Other": ["string"]
          }
      }`;

    const response = await openai.chat.completions.create({
      model: "gpt-3.5-turbo-16k",
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: `Parse the following resume:\n\n${text}` },
      ],
      temperature: 0.3,
      max_tokens: 4000,
    });

    const parsedResume = JSON.parse(response.choices[0].message.content);
    console.log("[Parse] Successfully parsed resume");

    res.json({ result: parsedResume });
  } catch (error) {
    console.error("[Parse] Error:", error);
    res.status(500).json({ error: "Failed to parse resume" });
  }
});

function normalizeSkill(skill) {
  skill = skill.toLowerCase().trim();

  const variations = {
    javascript: ["javascript", "js", "es6"],
    typescript: ["typescript", "ts"],
    python: ["python", "python3"],
    react: ["react", "reactjs", "react.js"],
    node: ["node", "nodejs", "node.js"],
    postgresql: ["postgresql", "postgres", "psql"],
    aws: ["aws", "amazon web services"],
    gcp: ["gcp", "google cloud platform"],
    cicd: ["ci/cd", "continuous integration/continuous delivery"],
    html: ["html", "html5"],
  };

  for (const [baseSkill, variants] of Object.entries(variations)) {
    if (variants.includes(skill)) {
      return baseSkill;
    }
  }

  return skill;
}

// Function to call Python ML code
async function callPythonML(resume_content, job_description) {
  console.log("[ML] Calling ml_resume.py for analysis");

  return new Promise((resolve, reject) => {
    const pythonProcess = spawn("python", [
      path.join(__dirname, "python/ml_resume.py"),
      "--resume_json",
      JSON.stringify(resume_content),
      "--job_description",
      job_description,
    ]);

    let mlOutput = "";

    pythonProcess.stdout.on("data", (data) => {
      mlOutput += data.toString();
      console.log("[ML] Python Output:", data.toString());
    });

    pythonProcess.stderr.on("data", (data) => {
      console.error("[ML] Python Error:", data.toString());
    });

    pythonProcess.on("close", (code) => {
      if (code !== 0) {
        console.error("[ML] Python process exited with code", code);
        reject(new Error("ML analysis failed"));
        return;
      }

      try {
        const mlResult = JSON.parse(mlOutput);
        console.log("[ML] Analysis completed:", mlResult);
        resolve(mlResult);
      } catch (error) {
        reject(error);
      }
    });
  });
}

app.post("/analyze-match", async (req, res) => {
  try {
    const { resume_content, job_description } = req.body;

    // Call Python ML code for analysis
    const mlResult = await callPythonML(resume_content, job_description);
    console.log("[ML] Features from Python ML:", mlResult);

    const mlScore = mlResult.prediction;
    console.log("[ML] Final score from Python ML model:", mlScore);

    // Get GenAI Analysis
    const genAIResponse = await openai.chat.completions.create({
      model: "gpt-3.5-turbo",
      messages: [
        {
          role: "system",
          content: `Analyze the technical skills match between the resume and job description.
          In your analysis, treat common skill variations and combined skills as equivalent. Specifically:
          - Treat 'React.js' and 'Reactjs' as the same skill.
          - Treat 'HTML5' and 'HTML' as the same skill.
          - Treat 'Node.js' and 'Nodejs' as the same skill.
          - Treat 'AWS' and 'Amazon Web Services' as the same skill.
          - Treat 'GCP' and 'Google Cloud Platform' as the same skill.
          - Treat 'CI/CD' and 'Continuous Integration/Continuous Delivery' as the same skill.
          - If skills are listed together in a 'slash' format (e.g., 'Typescript/Javascript'), treat each skill individually.
          
          Return a JSON with:
          {
            "score": number (0-100),
            "matched_skills": {
              "required": ["skill1"],
              "preferred": ["skill2"]
            },
            "missing_skills": {
              "required": ["skill3"],
              "preferred": ["skill4"]
            }
          }`,
        },
        {
          role: "user",
          content: `Resume Skills:\n${JSON.stringify(
            resume_content.skills
          )}\n\nJob Requirements:\n${job_description}`,
        },
      ],
      temperature: 0.1,
    });

    const genAIAnalysis = JSON.parse(genAIResponse.choices[0].message.content);
    const genAIScore = genAIAnalysis.score;

    // Combine ML and GenAI scores (30% ML, 70% GenAI)
    const combinedTechnicalScore = mlScore * 0.3 + genAIScore * 0.7;

    const expResponse = await openai.chat.completions.create({
      model: "gpt-3.5-turbo",
      messages: [
        {
          role: "system",
          content:
            "Extract the required years of experience from the job description. Return only the number.",
        },
        { role: "user", content: job_description },
      ],
      temperature: 0.1,
    });

    const requiredExp =
      parseFloat(expResponse.choices[0].message.content.trim()) || 0;

    const resumeExp = resume_content.work_experience.reduce((total, exp) => {
      if (exp.start_date && exp.end_date !== "present") {
        const startYear = new Date(exp.start_date).getFullYear();
        const endYear =
          exp.end_date === "present"
            ? new Date().getFullYear()
            : new Date(exp.end_date).getFullYear();
        return total + (endYear - startYear);
      }
      return total;
    }, 0);

    const experienceScore = Math.min(
      100,
      (resumeExp / Math.max(1, requiredExp)) * 100
    );

    const overallScore = Math.round(
      combinedTechnicalScore * 0.7 + experienceScore * 0.3
    );

    // Final response with both ML and GenAI components
    const response = {
      job_analysis: {
        match_score: {
          overall_percentage: overallScore,
          technical_skills_match: Math.round(combinedTechnicalScore),
          experience_match: Math.round(experienceScore),
          ml_components: {
            score: Math.round(mlScore),
            features: mlResult.features,
            model_info: "Using ml_resume.py with RandomForest model",
          },
          genai_score: genAIScore,
        },
        analysis: {
          matched_skills: genAIAnalysis.matched_skills,
          missing_skills: genAIAnalysis.missing_skills,
          experience: {
            has_years: resumeExp,
            required_years: requiredExp,
          },
        },
      },
    };

    console.log("[Analyze] Analysis completed successfully");
    res.json(response);
  } catch (error) {
    console.error("[Analyze] Error:", error);
    res.status(500).json({ error: "Failed to analyze job match" });
  }
});

function calculateMLScore(resume, jobDescription) {
  const resumeSkills = new Set(
    Object.values(resume.skills)
      .flat()
      .map((skill) => normalizeSkill(skill))
  );

  const jobSkills = extractSkillsFromJobDescription(jobDescription);
  const requiredSkills = new Set(
    jobSkills.required.map((skill) => normalizeSkill(skill))
  );
  const preferredSkills = new Set(
    jobSkills.preferred.map((skill) => normalizeSkill(skill))
  );

  const requiredMatches = [...requiredSkills].filter((skill) =>
    resumeSkills.has(skill)
  ).length;
  const preferredMatches = [...preferredSkills].filter((skill) =>
    resumeSkills.has(skill)
  ).length;

  const requiredScore = requiredSkills.size
    ? (requiredMatches / requiredSkills.size) * 70
    : 70;
  const preferredScore = preferredSkills.size
    ? (preferredMatches / preferredSkills.size) * 30
    : 30;

  return Math.round(requiredScore + preferredScore);
}

function extractSkillsFromJobDescription(jobDescription) {
  const text = jobDescription.toLowerCase();
  const skills = {
    required: [],
    preferred: [],
  };

  const commonTechSkills = [
    "javascript",
    "python",
    "java",
    "react",
    "node",
    "aws",
    "docker",
    "kubernetes",
    "sql",
    "nosql",
    "mongodb",
    "postgresql",
    "git",
    "ci/cd",
    "typescript",
    "angular",
    "vue",
    "graphql",
    "rest",
    "api",
    "cloud",
    "microservices",
    "agile",
    "scrum",
    "devops",
    "testing",
    "html",
    "css",
  ];

  commonTechSkills.forEach((skill) => {
    if (text.includes(skill)) {
      if (
        text.includes(`must have ${skill}`) ||
        text.includes(`required: ${skill}`)
      ) {
        skills.required.push(skill);
      } else {
        skills.preferred.push(skill);
      }
    }
  });

  return skills;
}

exports.api = functions.runWith(runtimeOpts).https.onRequest(app);
