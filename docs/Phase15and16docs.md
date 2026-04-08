# Phase 15 and 16 Notes

## Overview

Phase 15 adds the structured generative report layer on top of the existing
Phase 10 and Phase 12 pipeline. Phase 16 adds explainability, confidence
calibration, fairness checks, and audit metadata.

The Phase 15/16 API surface is available under `/api/ai`.

## Endpoints

### Generate

- `POST /api/ai/generate/student-report`
- `POST /api/ai/generate/professor-report`

These return the frontend-oriented tab structure:

- `prediction`
- `explanation`
- `sources`
- `warnings`
- `fairness`
- `audit`

### Read-only inspection

- `GET /api/ai/explain/{file_id}`
- `GET /api/ai/compare/{file_id}`
- `GET /api/ai/audit/{file_id}`
- `GET /api/ai/pdf/{file_id}`

The PDF endpoint returns a generated `application/pdf` attachment for the most
recent Phase 15/16 report associated with the requested file and role.

## Compare Endpoint

`GET /api/ai/compare/{file_id}` returns the fairness payload plus compact
comparison metadata:

- `comparison_summary`
- `confidence_score`
- `confidence_band`
- `evidence_reference_count`
- `warning_count`

This is intended for lightweight UI comparison panels rather than full audit
inspection.

## Confidence Thresholds

Phase 15/16 confidence bands are configurable through `GENAI_*` settings in
[backend/app/genai/config.py](/c:/Users/sahua/OneDrive/Desktop/ANSH/EduAIPlatform/backend/app/genai/config.py).

Current defaults:

- `GENAI_CONFIDENCE_HIGH_THRESHOLD=0.75`
- `GENAI_CONFIDENCE_MEDIUM_THRESHOLD=0.45`

Band mapping:

- `score >= high threshold` -> `high`
- `score >= medium threshold and < high threshold` -> `medium`
- `score < medium threshold` -> `low`

These thresholds are used by the explainability layer and any response fields
that expose calibrated confidence bands.

## PDF Notes

PDF generation uses ReportLab and is enabled by `GENAI_PDF_ENABLED=true`.
The PDF contains:

- report summary
- confidence band and score
- strengths
- weaknesses
- suggestions
- compact evidence references

If ReportLab is unavailable locally, the backend still starts successfully, but
PDF generation will fail at request time rather than startup time.
