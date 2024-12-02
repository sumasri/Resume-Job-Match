import { useState } from "react";
import pdfToText from "react-pdftotext";
import {
  Container,
  Grid,
  Button,
  TextField,
  Paper,
  Typography,
  Box,
  CircularProgress,
  Chip,
  ThemeProvider,
  createTheme,
  alpha,
} from "@mui/material";
import axios from "axios";

const theme = createTheme({
  palette: {
    primary: {
      main: "#2563eb",
      light: "#3b82f6",
      dark: "#1d4ed8",
    },
    secondary: {
      main: "#64748b",
      light: "#94a3b8",
      dark: "#475569",
    },
    success: {
      main: "#059669",
      light: "#10b981",
      dark: "#047857",
    },
    error: {
      main: "#dc2626",
      light: "#ef4444",
      dark: "#b91c1c",
    },
    warning: {
      main: "#d97706",
      light: "#f59e0b",
      dark: "#b45309",
    },
  },
  typography: {
    fontFamily: "'Inter', sans-serif",
    h4: {
      fontWeight: 700,
    },
    h6: {
      fontWeight: 600,
    },
  },
  shape: {
    borderRadius: 12,
  },
  components: {
    MuiPaper: {
      styleOverrides: {
        root: {
          boxShadow:
            "0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1)",
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: "none",
          fontWeight: 600,
          padding: "10px 24px",
        },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: {
          fontWeight: 500,
        },
      },
    },
  },
});

const API_URL = "https://us-central1-lma-project-15974.cloudfunctions.net/api";

function App() {
  const [resume, setResume] = useState(null);
  const [jobDescription, setJobDescription] = useState("");
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleResumeUpload = (event) => {
    const file = event.target.files[0];
    setResume(file);
  };

  const handleReset = () => {
    setResume(null);
    setJobDescription("");
    setAnalysis(null);
  };

  const extractTextFromPDF = async (file) => {
    try {
      const text = await pdfToText(file);
      const formattedText = text
        .replace(/\r\n/g, "\n")
        .replace(/\s+/g, " ")
        .replace(/\n+/g, "\n")
        .trim();
      const sections = formattedText
        .split("\n\n")
        .map((section) => section.trim());
      const structuredText = sections.join("\n\n");
      console.log("Extracted text with formatting:", {
        originalLength: text.length,
        formattedLength: structuredText.length,
        sections: sections.length,
      });
      return structuredText;
    } catch (error) {
      console.error("Error extracting text from PDF:", error);
      throw error;
    }
  };

  const handleAnalyze = async () => {
    if (!resume || !jobDescription) {
      alert("Please upload a resume and enter job description");
      return;
    }

    setLoading(true);
    try {
      const resumeText = await extractTextFromPDF(resume);
      console.log("Sample of extracted text:", resumeText.substring(0, 500));

      const parseResponse = await axios.post(`${API_URL}/parse`, {
        text: resumeText,
        filename: resume.name,
      });

      const parsedResumeData = parseResponse.data.result;

      const analyzeResponse = await axios.post(`${API_URL}/analyze-match`, {
        resume_content: parsedResumeData,
        job_description: jobDescription,
      });

      setAnalysis(analyzeResponse.data.job_analysis);
    } catch (error) {
      console.error("Error:", error);
      alert("An error occurred during analysis");
    } finally {
      setLoading(false);
    }
  };

  const renderAnalysisResults = () => {
    if (!analysis) return null;

    const { match_score, analysis: details } = analysis;

    const matchedSkills = {
      required: details?.matched_skills?.required || [],
      preferred: details?.matched_skills?.preferred || [],
    };

    const missingSkills = {
      required: details?.missing_skills?.required || [],
      preferred: details?.missing_skills?.preferred || [],
    };

    const experience = {
      has_years: details?.experience?.has_years || 0,
      required_years: details?.experience?.required_years || 0,
    };

    return (
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Paper
            sx={{
              p: 4,
              background: "linear-gradient(to bottom right, #ffffff, #f8fafc)",
              height: "100%",
            }}
          >
            <Box sx={{ mb: 4 }}>
              <Typography
                variant="h6"
                gutterBottom
                sx={{ color: "primary.dark" }}
              >
                Match Scores
              </Typography>
              <Grid container spacing={2}>
                {[
                  {
                    label: "Overall Match",
                    value: match_score.overall_percentage,
                    icon: "🎯",
                  },
                  {
                    label: "Technical Skills",
                    value: match_score.technical_skills_match,
                    icon: "💻",
                  },
                  {
                    label: "Experience Match",
                    value: match_score.experience_match,
                    icon: "⭐",
                  },
                ].map((score) => (
                  <Grid item xs={4} key={score.label}>
                    <Paper
                      elevation={0}
                      sx={{
                        p: 2,
                        textAlign: "center",
                        bgcolor:
                          score.value >= 70
                            ? alpha(theme.palette.success.main, 0.1)
                            : alpha(theme.palette.error.main, 0.1),
                        border: 1,
                        borderColor:
                          score.value >= 70 ? "success.light" : "error.light",
                      }}
                    >
                      <Typography variant="h5" sx={{ mb: 1 }}>
                        {score.icon}
                      </Typography>
                      <Typography
                        variant="h4"
                        sx={{
                          color:
                            score.value >= 70 ? "success.main" : "error.main",
                          fontWeight: "bold",
                        }}
                      >
                        {score.value}%
                      </Typography>
                      <Typography
                        variant="subtitle2"
                        sx={{ color: "text.secondary" }}
                      >
                        {score.label}
                      </Typography>
                    </Paper>
                  </Grid>
                ))}
              </Grid>
            </Box>

            <Box>
              <Typography
                variant="h6"
                gutterBottom
                sx={{ color: "primary.dark" }}
              >
                Experience Analysis
              </Typography>
              <Paper
                elevation={0}
                sx={{
                  p: 3,
                  bgcolor: alpha(theme.palette.primary.main, 0.05),
                  border: 1,
                  borderColor: "primary.light",
                }}
              >
                <Typography variant="body1" sx={{ lineHeight: 1.6 }}>
                  You have <strong>{experience.has_years} years</strong> of
                  experience, while the job requires{" "}
                  <strong>{experience.required_years} years</strong>.
                </Typography>
              </Paper>
            </Box>
          </Paper>
        </Grid>

        <Grid item xs={12} md={6}>
          <Paper
            sx={{
              p: 4,
              background: "linear-gradient(to bottom right, #ffffff, #f8fafc)",
              height: "100%",
            }}
          >
            <Typography
              variant="h6"
              gutterBottom
              sx={{ color: "primary.dark" }}
            >
              Skills Analysis
            </Typography>

            <Box sx={{ mb: 4 }}>
              <Typography
                variant="subtitle2"
                color="success.main"
                gutterBottom
                sx={{ fontWeight: 600 }}
              >
                Matched Skills
              </Typography>
              <Paper
                elevation={0}
                sx={{
                  p: 3,
                  bgcolor: alpha(theme.palette.success.main, 0.05),
                  border: 1,
                  borderColor: "success.light",
                }}
              >
                <Box sx={{ mb: 2 }}>
                  <Typography variant="body2" gutterBottom>
                    Required:
                  </Typography>
                  <Box
                    sx={{ display: "flex", gap: 1, flexWrap: "wrap", mb: 1 }}
                  >
                    {matchedSkills.required.map((skill) => (
                      <Chip
                        key={skill}
                        label={skill}
                        color="success"
                        variant="outlined"
                        size="small"
                      />
                    ))}
                  </Box>
                </Box>
                <Box>
                  <Typography variant="body2" gutterBottom>
                    Preferred:
                  </Typography>
                  <Box sx={{ display: "flex", gap: 1, flexWrap: "wrap" }}>
                    {matchedSkills.preferred.map((skill) => (
                      <Chip
                        key={skill}
                        label={skill}
                        color="primary"
                        variant="outlined"
                        size="small"
                      />
                    ))}
                  </Box>
                </Box>
              </Paper>
            </Box>

            <Box>
              <Typography
                variant="subtitle2"
                color="error.main"
                gutterBottom
                sx={{ fontWeight: 600 }}
              >
                Missing Skills
              </Typography>
              <Paper
                elevation={0}
                sx={{
                  p: 3,
                  bgcolor: alpha(theme.palette.error.main, 0.05),
                  border: 1,
                  borderColor: "error.light",
                }}
              >
                {missingSkills.required.length > 0 && (
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="body2" gutterBottom>
                      Required:
                    </Typography>
                    <Box
                      sx={{ display: "flex", gap: 1, flexWrap: "wrap", mb: 1 }}
                    >
                      {missingSkills.required.map((skill) => (
                        <Chip
                          key={skill}
                          label={skill}
                          color="error"
                          variant="outlined"
                          size="small"
                        />
                      ))}
                    </Box>
                  </Box>
                )}
                <Box>
                  <Typography variant="body2" gutterBottom>
                    Preferred:
                  </Typography>
                  <Box sx={{ display: "flex", gap: 1, flexWrap: "wrap" }}>
                    {missingSkills.preferred.map((skill) => (
                      <Chip
                        key={skill}
                        label={skill}
                        color="warning"
                        variant="outlined"
                        size="small"
                      />
                    ))}
                  </Box>
                </Box>
              </Paper>
            </Box>
          </Paper>
        </Grid>
      </Grid>
    );
  };

  return (
    <ThemeProvider theme={theme}>
      <Box
        sx={{
          minHeight: "100vh",
          width: "100%",
          bgcolor: "#f1f5f9",
          position: "fixed",
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          overflowY: "auto",
        }}
      >
        <Container
          maxWidth="lg"
          sx={{
            py: 6,
            height: "100%",
            display: "flex",
            flexDirection: "column",
          }}
        >
          <Typography
            variant="h6"
            gutterBottom
            align="center"
            sx={{
              mb: 4,
              background: "linear-gradient(45deg, #1e40af, #3b82f6)",
              backgroundClip: "text",
              textFillColor: "transparent",
              WebkitBackgroundClip: "text",
              WebkitTextFillColor: "transparent",
            }}
          >
            Smart Resume Job Matching: A Hybrid Approach Using Generative AI and
            ML for Scoring and Insights
          </Typography>

          {!analysis ? (
            <>
              <Grid container spacing={3}>
                <Grid item xs={12} md={6}>
                  <Paper
                    sx={{
                      p: 4,
                      height: "100%",
                      background:
                        "linear-gradient(to bottom right, #ffffff, #f8fafc)",
                    }}
                  >
                    <Typography
                      variant="h6"
                      gutterBottom
                      sx={{ color: "primary.dark" }}
                    >
                      Upload Resume
                    </Typography>
                    <Button
                      component="label"
                      variant="outlined"
                      sx={{
                        width: "100%",
                        height: 100,
                        borderStyle: "dashed",
                        borderWidth: 2,
                      }}
                    >
                      {resume ? resume.name : "Choose PDF, DOC, or DOCX file"}
                      <input
                        type="file"
                        hidden
                        accept=".pdf,.doc,.docx"
                        onChange={handleResumeUpload}
                      />
                    </Button>
                  </Paper>
                </Grid>

                <Grid item xs={12} md={6}>
                  <Paper
                    sx={{
                      p: 4,
                      height: "100%",
                      background:
                        "linear-gradient(to bottom right, #ffffff, #f8fafc)",
                    }}
                  >
                    <Typography
                      variant="h6"
                      gutterBottom
                      sx={{ color: "primary.dark" }}
                    >
                      Job Description
                    </Typography>
                    <TextField
                      fullWidth
                      multiline
                      rows={6}
                      value={jobDescription}
                      onChange={(e) => setJobDescription(e.target.value)}
                      placeholder="Paste job description here..."
                      variant="outlined"
                      sx={{
                        "& .MuiOutlinedInput-root": {
                          bgcolor: "white",
                        },
                      }}
                    />
                  </Paper>
                </Grid>
              </Grid>

              <Box sx={{ mt: 4, textAlign: "center" }}>
                <Button
                  variant="contained"
                  onClick={handleAnalyze}
                  disabled={loading}
                  size="large"
                  sx={{
                    px: 6,
                    py: 1.5,
                    borderRadius: 3,
                    background: "linear-gradient(45deg, #1e40af, #3b82f6)",
                    minWidth: 200,
                    position: "relative",
                  }}
                >
                  {loading ? (
                    <CircularProgress
                      size={24}
                      sx={{
                        color: "white",
                        position: "absolute",
                        left: "50%",
                        marginLeft: "-12px",
                      }}
                    />
                  ) : (
                    "Analyze Match"
                  )}
                </Button>
              </Box>
            </>
          ) : (
            <>
              {renderAnalysisResults()}
              <Box sx={{ mt: 4, textAlign: "center" }}>
                <Button
                  variant="contained"
                  onClick={handleReset}
                  size="large"
                  color="secondary"
                  sx={{
                    px: 6,
                    py: 1.5,
                    borderRadius: 3,
                    minWidth: 200,
                    position: "relative",
                  }}
                >
                  Analyze Another Resume
                </Button>
              </Box>
            </>
          )}

          <Box
            sx={{
              mt: "auto",
              pt: 4,
              textAlign: "center",
              opacity: 0.7,
            }}
          >
            <Typography variant="body2" color="text.secondary">
              Powered by: Let me apply
            </Typography>
          </Box>
        </Container>
      </Box>
    </ThemeProvider>
  );
}

export default App;
